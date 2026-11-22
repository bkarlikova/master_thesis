(* Assigns every tensor a fixed byte offset in WASM linear memory.
   Simplest strategy: one region per tensor, laid out one after another,
   never reused. The input and output come first so they are easy to find. *)

type t = { offsets : (string * int) list; total_bytes : int }

let bytes_per_float = 4

let plan (graph : Ir.graph) : t =
  let others =
    List.filter
      (fun (t : Ir.tensor) -> t.name <> graph.input && t.name <> graph.output)
      graph.tensors
  in
  let ordered =
    Ir.find_tensor graph graph.input :: Ir.find_tensor graph graph.output :: others
  in
  let offsets, total_bytes =
    List.fold_left
      (fun (offsets, next) (t : Ir.tensor) ->
        let size = Ir.element_count t.shape * bytes_per_float in
        (offsets @ [ (t.name, next) ], next + size))
      ([], 0) ordered
  in
  { offsets; total_bytes }

let offset (plan : t) (name : string) : int =
  match List.assoc_opt name plan.offsets with
  | Some offset -> offset
  | None -> failwith ("tensor has no memory region: " ^ name)

let print (graph : Ir.graph) (plan : t) : unit =
  Printf.printf "Memory plan (%d bytes):\n" plan.total_bytes;
  List.iter
    (fun (name, offset) ->
      let t = Ir.find_tensor graph name in
      Printf.printf "  %6d  %-14s %d bytes\n" offset name
        (Ir.element_count t.shape * bytes_per_float))
    plan.offsets
