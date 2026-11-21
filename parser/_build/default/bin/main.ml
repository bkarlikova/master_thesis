open Onnx_proto.Onnx
open Onnx

let read_file_bytes (path : string) : string =
  In_channel.with_open_bin path In_channel.input_all

let str (o : string option) : string = Option.value o ~default:""
let i32 (o : int option) : int = Option.value o ~default:0

let print_tensor_shape (shape_opt : Onnx.TensorShapeProto.t option) : string =
  match shape_opt with
  | None -> "unknown shape"
  | Some shape ->
    let dims =
      List.map (fun (d : Onnx.TensorShapeProto.Dimension.t) ->
        match d.value with
        | `Dim_value v -> string_of_int v
        | `Dim_param p -> p
        | `not_set -> "?") shape
    in
    "[" ^ String.concat ", " dims ^ "]"

let inspect_graph (graph : GraphProto.t) =
  Printf.printf "=== Graph: %s ===\n" (str graph.name);

  (* 1. Graph Inputs *)
  Printf.printf "\n[Inputs (%d)]:\n" (List.length graph.input);
  List.iter (fun (vi : ValueInfoProto.t) ->
    let shape_str =
      match vi.type' with
      | Some { value = `Tensor_type t; _ } -> print_tensor_shape t.shape
      | _ -> "non-tensor"
    in
    Printf.printf "  - %s : %s\n" (str vi.name) shape_str
  ) graph.input;

  (* 2. Graph Initializers (Constant Weights & Biases) *)
  Printf.printf "\n[Initializers / Weights (%d)]:\n" (List.length graph.initializer');
  List.iter (fun (t : TensorProto.t) ->
    let dims_str = String.concat "x" (List.map string_of_int t.dims) in
    Printf.printf "  - %s : shape [%s], data_type: %d\n"
      (str t.name) dims_str (i32 t.data_type)
  ) graph.initializer';

  (* 3. Computation Nodes (Execution Pipeline) *)
  Printf.printf "\n[Topological Nodes (%d)]:\n" (List.length graph.node);
  List.iteri (fun idx (n : NodeProto.t) ->
    let inputs = String.concat ", " n.input in
    let outputs = String.concat ", " n.output in
    let op_name = match n.name with None | Some "" -> "<unnamed>" | Some s -> s in
    Printf.printf "  %3d. %s (%s)\n" idx (str n.op_type) op_name;
    Printf.printf "       In:  [%s]\n" inputs;
    Printf.printf "       Out: [%s]\n" outputs;
  ) graph.node;

  (* 4. Graph Outputs *)
  Printf.printf "\n[Outputs (%d)]:\n" (List.length graph.output);
  List.iter (fun (vi : ValueInfoProto.t) ->
    Printf.printf "  - %s\n" (str vi.name)
  ) graph.output

let parse_and_inspect (filepath : string) =
  Printf.printf "Loading binary ONNX model from: %s\n" filepath;
  let raw_bytes = read_file_bytes filepath in
  let reader = Ocaml_protoc_plugin.Reader.create raw_bytes in

  match ModelProto.from_proto reader with
  | Ok model ->
    Printf.printf "Successfully parsed ONNX Model!\n";
    Printf.printf "IR Version: %d\n" (i32 model.ir_version);
    Printf.printf "Producer: %s %s\n" (str model.producer_name) (str model.producer_version);

    (* Print operator set versions *)
    List.iter (fun (opset : OperatorSetIdProto.t) ->
      let domain =
        match opset.domain with
        | None | Some "" -> "ai.onnx (default)"
        | Some d -> d
      in
      Printf.printf "Opset Import: %s -> v%d\n" domain (i32 opset.version)
    ) model.opset_import;

    (* Inspect the inner computation DAG *)
    (match model.graph with
     | Some g -> inspect_graph g
     | None -> Printf.printf "Model contains no graph payload.\n")
  | Error err ->
    Printf.eprintf "Failed to decode ONNX model: %s\n"
      (Ocaml_protoc_plugin.Result.show_error err);
    exit 1

let () =
  if Array.length Sys.argv < 2 then (
    Printf.eprintf "Usage: %s <path_to_model.onnx>\n" Sys.argv.(0);
    exit 1
  );
  parse_and_inspect Sys.argv.(1)