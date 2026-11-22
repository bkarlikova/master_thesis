(* Turns an Ir.graph plus its memory plan into a WASM module with one
   exported function, run, that executes all nodes in order. *)

open Wasm

(* Locals of run. Each op's loops reuse the same counters. *)
let local_row = 0
let local_column = 1
let local_inner = 2
let local_accumulator = 3
let run_locals = [ I32; I32; I32; F32 ]

(* for counter = 0 to count - 1 do body done.
   Written as a do-while loop; correct because every dimension is >= 1. *)
let for_loop (counter : int) (count : int) (body : instr list) : instr list =
  [
    I32_const 0;
    Local_set counter;
    Loop
      (body
      @ [
          Local_get counter; I32_const 1; I32_add; Local_set counter;
          Local_get counter; I32_const count; I32_lt_s; Br_if 0;
        ]);
  ]

(* Byte address of element [row, column] of a row-major matrix, relative to
   the matrix's own offset (which goes into the load/store instruction). *)
let element_address ~(row : int) ~(column : int) ~(columns : int) : instr list =
  [
    Local_get row; I32_const columns; I32_mul; Local_get column; I32_add;
    I32_const Memory_plan.bytes_per_float; I32_mul;
  ]

let matrix_shape (graph : Ir.graph) (name : string) : int * int =
  match (Ir.find_tensor graph name).shape with
  | [ rows; columns ] -> (rows, columns)
  | _ -> failwith ("expected a matrix: " ^ name)

(* Y[m, n] = alpha * sum_k A[m, k] * B[k, n] + beta * C[n]
   With transA/transB the same element is read at the transposed position. *)
let gemm graph plan ~alpha ~beta ~trans_a ~trans_b inputs output : instr list =
  let a, b, bias =
    match inputs with
    | [ a; b ] -> (a, b, None)
    | [ a; b; c ] -> (a, b, Some c)
    | _ -> failwith "Gemm expects 2 or 3 inputs"
  in
  let _, a_columns = matrix_shape graph a in
  let _, b_columns = matrix_shape graph b in
  let rows, columns = matrix_shape graph output in
  let inner = if trans_a then fst (matrix_shape graph a) else a_columns in
  let a_element =
    if trans_a then element_address ~row:local_inner ~column:local_row ~columns:a_columns
    else element_address ~row:local_row ~column:local_inner ~columns:a_columns
  in
  let b_element =
    if trans_b then element_address ~row:local_column ~column:local_inner ~columns:b_columns
    else element_address ~row:local_inner ~column:local_column ~columns:b_columns
  in
  let offset = Memory_plan.offset plan in
  let multiply_accumulate =
    [ Local_get local_accumulator ]
    @ a_element @ [ F32_load (offset a) ]
    @ b_element @ [ F32_load (offset b); F32_mul; F32_add; Local_set local_accumulator ]
  in
  let add_bias =
    match bias with
    | None -> []
    | Some c ->
      [
        Local_get local_column; I32_const Memory_plan.bytes_per_float; I32_mul;
        F32_load (offset c); F32_const beta; F32_mul; F32_add;
      ]
  in
  let store_result =
    element_address ~row:local_row ~column:local_column ~columns
    @ [ Local_get local_accumulator; F32_const alpha; F32_mul ]
    @ add_bias
    @ [ F32_store (offset output) ]
  in
  for_loop local_row rows
    (for_loop local_column columns
       ([ F32_const 0.0; Local_set local_accumulator ]
       @ for_loop local_inner inner multiply_accumulate
       @ store_result))

(* Y[i] = max(X[i], 0) over the flattened tensor. *)
let relu graph plan input output : instr list =
  let count = Ir.element_count (Ir.find_tensor graph input).shape in
  let address = [ Local_get local_row; I32_const Memory_plan.bytes_per_float; I32_mul ] in
  for_loop local_row count
    (address @ address
    @ [
        F32_load (Memory_plan.offset plan input); F32_const 0.0; F32_max;
        F32_store (Memory_plan.offset plan output);
      ])

let compile_node graph plan (node : Ir.node) : instr list =
  match node.op, node.inputs with
  | Gemm { alpha; beta; trans_a; trans_b }, inputs ->
    gemm graph plan ~alpha ~beta ~trans_a ~trans_b inputs node.output
  | Relu, [ input ] -> relu graph plan input node.output
  | Relu, _ -> failwith "Relu expects 1 input"

let weights_as_bytes (weights : float array) : string =
  let buf = Buffer.create (Array.length weights * Memory_plan.bytes_per_float) in
  Array.iter (fun w -> Buffer.add_int32_le buf (Int32.bits_of_float w)) weights;
  Buffer.contents buf

let page_size = 65536

let compile (graph : Ir.graph) (plan : Memory_plan.t) : Wasm.module_ =
  let body = List.concat_map (compile_node graph plan) graph.nodes in
  let data =
    List.filter_map
      (fun (t : Ir.tensor) ->
        Option.map
          (fun weights -> (Memory_plan.offset plan t.name, weights_as_bytes weights))
          t.weights)
      graph.tensors
  in
  let length name = Ir.element_count (Ir.find_tensor graph name).shape in
  {
    memory_pages = max 1 ((plan.total_bytes + page_size - 1) / page_size);
    functions = [ { export_name = "run"; locals = run_locals; body } ];
    constants =
      [
        ("input_offset", Memory_plan.offset plan graph.input);
        ("input_length", length graph.input);
        ("output_offset", Memory_plan.offset plan graph.output);
        ("output_length", length graph.output);
      ];
    data;
  }
