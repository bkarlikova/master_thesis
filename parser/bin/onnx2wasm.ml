(* onnx2wasm <model.onnx> <output.wasm> [batch]
   Compiles an ONNX model into a standalone WASM module and prints the
   intermediate representation and memory plan for inspection. *)

open Onnx2wasm

let () =
  let model_path, output_path, batch =
    match Array.to_list Sys.argv with
    | [ _; model; output ] -> (model, output, 1)
    | [ _; model; output; batch ] -> (model, output, int_of_string batch)
    | _ ->
      prerr_endline "Usage: onnx2wasm <model.onnx> <output.wasm> [batch]";
      exit 2
  in
  match
    let graph = Loader.load ~batch model_path in
    let plan = Memory_plan.plan graph in
    Ir.print graph;
    Memory_plan.print graph plan;
    Wasm.encode (Codegen.compile graph plan)
  with
  | bytes ->
    Out_channel.with_open_bin output_path (fun channel ->
        Out_channel.output_string channel bytes);
    Printf.printf "Wrote %s (%d bytes)\n" output_path (String.length bytes)
  | exception Failure message ->
    Printf.eprintf "onnx2wasm: %s\n" message;
    exit 1
