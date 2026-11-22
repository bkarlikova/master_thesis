(* Reads an ONNX file into Ir.graph. Fails with a clear message on anything
   outside the supported subset, so unsupported models are never compiled
   silently into something wrong. *)

open Onnx_proto.Onnx.Onnx

let onnx_float = 1 (* TensorProto.DataType.FLOAT *)

let read_file (path : string) : string =
  In_channel.with_open_bin path In_channel.input_all

let floats_of_little_endian_bytes (bytes : string) : float array =
  if String.length bytes mod 4 <> 0 then
    failwith "tensor byte length is not a multiple of 4";
  Array.init (String.length bytes / 4) (fun i ->
      Int32.float_of_bits (String.get_int32_le bytes (4 * i)))

let external_entry (tensor : TensorProto.t) (key : string) : string option =
  List.find_map
    (fun (entry : StringStringEntryProto.t) ->
      if entry.key = Some key then entry.value else None)
    tensor.external_data

(* External weights live in a side file next to the model, described by
   location/offset/length entries. *)
let read_external_bytes ~(model_dir : string) (tensor : TensorProto.t) : string =
  let location =
    match external_entry tensor "location" with
    | Some location -> location
    | None -> failwith "external tensor without a location"
  in
  let data = read_file (Filename.concat model_dir location) in
  let offset =
    Option.fold ~none:0 ~some:int_of_string (external_entry tensor "offset")
  in
  let length =
    Option.fold ~none:(String.length data - offset) ~some:int_of_string
      (external_entry tensor "length")
  in
  String.sub data offset length

let load_initializer ~(model_dir : string) (tensor : TensorProto.t) : Ir.tensor =
  let name = Option.value tensor.name ~default:"" in
  if tensor.data_type <> Some onnx_float then
    failwith ("only float32 weights are supported: " ^ name);
  let weights =
    match tensor.data_location, tensor.raw_data, tensor.float_data with
    | Some TensorProto.DataLocation.EXTERNAL, _, _ ->
      floats_of_little_endian_bytes (read_external_bytes ~model_dir tensor)
    | _, Some raw, _ -> floats_of_little_endian_bytes (Bytes.to_string raw)
    | _, None, values -> Array.of_list values
  in
  let shape = tensor.dims in
  if Array.length weights <> Ir.element_count shape then
    failwith ("weight count does not match shape: " ^ name);
  { name; shape; weights = Some weights }

(* The symbolic batch dimension (e.g. "batch_size") becomes [batch]. *)
let load_graph_input ~(batch : int) (value : ValueInfoProto.t) : Ir.tensor =
  let name = Option.value value.name ~default:"" in
  let dims =
    match value.type' with
    | Some { value = `Tensor_type { shape = Some dims; _ }; _ } -> dims
    | _ -> failwith ("input without a tensor shape: " ^ name)
  in
  let dim_size (dim : TensorShapeProto.Dimension.t) =
    match dim.value with
    | `Dim_value size -> size
    | `Dim_param _ -> batch
    | `not_set -> failwith ("input with an unknown dimension: " ^ name)
  in
  { name; shape = List.map dim_size dims; weights = None }

let attribute (node : NodeProto.t) (name : string) : AttributeProto.t option =
  List.find_opt
    (fun (attr : AttributeProto.t) -> attr.name = Some name)
    node.attribute

let float_attribute node name ~default =
  match attribute node name with
  | Some { f = Some value; _ } -> value
  | _ -> default

let bool_attribute node name =
  match attribute node name with
  | Some { i = Some value; _ } -> value <> 0
  | _ -> false

let convert_op (node : NodeProto.t) : Ir.op =
  match node.op_type with
  | Some "Gemm" ->
    Gemm
      {
        alpha = float_attribute node "alpha" ~default:1.0;
        beta = float_attribute node "beta" ~default:1.0;
        trans_a = bool_attribute node "transA";
        trans_b = bool_attribute node "transB";
      }
  | Some "Relu" -> Relu
  | Some other -> failwith ("unsupported operator: " ^ other)
  | None -> failwith "node without an operator type"

(* Output shape of a node, computed from the shapes of its inputs. *)
let infer_shape (op : Ir.op) (input_shapes : int list list) : int list =
  match op, input_shapes with
  | Relu, [ shape ] -> shape
  | Gemm { trans_a; trans_b; _ }, [ a0; a1 ] :: [ b0; b1 ] :: bias ->
    let m, k = if trans_a then (a1, a0) else (a0, a1) in
    let k_b, n = if trans_b then (b1, b0) else (b0, b1) in
    if k <> k_b then failwith "Gemm: inner dimensions do not match";
    (match bias with
     | [] -> ()
     | [ [ bias_n ] ] when bias_n = n -> ()
     | _ -> failwith "Gemm: only a bias of shape [N] is supported");
    [ m; n ]
  | _ -> failwith "operator applied to unexpected input shapes"

let load ~(batch : int) (path : string) : Ir.graph =
  let reader = Ocaml_protoc_plugin.Reader.create (read_file path) in
  let model =
    match ModelProto.from_proto reader with
    | Ok model -> model
    | Error err ->
      failwith
        ("cannot decode ONNX file: " ^ Ocaml_protoc_plugin.Result.show_error err)
  in
  let graph =
    match model.graph with
    | Some graph -> graph
    | None -> failwith "model has no graph"
  in
  let model_dir = Filename.dirname path in
  let weights = List.map (load_initializer ~model_dir) graph.initializer' in
  let is_weight name = List.exists (fun (t : Ir.tensor) -> t.name = name) weights in
  (* Older exporters also list weights as graph inputs; skip those. *)
  let inputs =
    List.filter
      (fun (v : ValueInfoProto.t) -> not (is_weight (Option.value v.name ~default:"")))
      graph.input
  in
  let input, output =
    match inputs, graph.output with
    | [ input ], [ output ] -> (load_graph_input ~batch input, output)
    | _ -> failwith "only models with one input and one output are supported"
  in
  (* Walk the nodes in order, adding each node's output tensor once its
     shape is known. *)
  let tensors, nodes =
    List.fold_left
      (fun (tensors, nodes) (proto : NodeProto.t) ->
        let op = convert_op proto in
        let shape_of name =
          match List.find_opt (fun (t : Ir.tensor) -> t.name = name) tensors with
          | Some t -> t.shape
          | None -> failwith ("node reads a tensor that is not defined yet: " ^ name)
        in
        let output =
          match proto.output with
          | [ output ] -> output
          | _ -> failwith "only nodes with one output are supported"
        in
        let shape = infer_shape op (List.map shape_of proto.input) in
        let tensor : Ir.tensor = { name = output; shape; weights = None } in
        ( tensors @ [ tensor ],
          nodes @ [ ({ op; inputs = proto.input; output } : Ir.node) ] ))
      (input :: weights, []) graph.node
  in
  {
    tensors;
    nodes;
    input = input.name;
    output = Option.value output.name ~default:"";
  }
