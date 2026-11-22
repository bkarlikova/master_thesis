(* The compiler's own view of a model: every shape is a concrete number
   (the batch size is fixed at compile time) and only supported ops exist. *)

type tensor = {
  name : string;
  shape : int list;
  weights : float array option;  (* Some for initializers, None for activations *)
}

type op =
  | Gemm of { alpha : float; beta : float; trans_a : bool; trans_b : bool }
  | Relu

type node = { op : op; inputs : string list; output : string }

type graph = {
  tensors : tensor list;  (* in definition order: inputs, weights, then node outputs *)
  nodes : node list;  (* in execution (topological) order *)
  input : string;
  output : string;
}

let element_count (shape : int list) : int = List.fold_left ( * ) 1 shape

let find_tensor (graph : graph) (name : string) : tensor =
  match List.find_opt (fun t -> t.name = name) graph.tensors with
  | Some tensor -> tensor
  | None -> failwith ("unknown tensor: " ^ name)

let shape_to_string (shape : int list) : string =
  "[" ^ String.concat "x" (List.map string_of_int shape) ^ "]"

let op_to_string (op : op) : string =
  match op with
  | Gemm { alpha; beta; trans_a; trans_b } ->
    Printf.sprintf "Gemm(alpha=%g, beta=%g, transA=%b, transB=%b)" alpha beta
      trans_a trans_b
  | Relu -> "Relu"

let print (graph : graph) : unit =
  Printf.printf "Tensors:\n";
  List.iter
    (fun t ->
      let kind = if Option.is_some t.weights then "weights" else "" in
      Printf.printf "  %-14s %-8s %s\n" t.name (shape_to_string t.shape) kind)
    graph.tensors;
  Printf.printf "Nodes:\n";
  List.iter
    (fun (n : node) ->
      Printf.printf "  %s = %s(%s)\n" n.output (op_to_string n.op)
        (String.concat ", " n.inputs))
    graph.nodes
