(* Unit tests for the WASM binary encoder. Expected bytes are taken from the
   LEB128 examples in the WebAssembly specification and on Wikipedia. *)

open Onnx2wasm

let hex (s : string) : string =
  String.concat " " (List.map (fun c -> Printf.sprintf "%02x" (Char.code c)) (String.to_seq s |> List.of_seq))

let check name encode value expected =
  let buf = Buffer.create 8 in
  encode buf value;
  let actual = hex (Buffer.contents buf) in
  if actual <> expected then (
    Printf.eprintf "FAIL %s %d: expected %s, got %s\n" name value expected actual;
    exit 1)

let () =
  check "unsigned" Wasm.unsigned_leb128 0 "00";
  check "unsigned" Wasm.unsigned_leb128 127 "7f";
  check "unsigned" Wasm.unsigned_leb128 128 "80 01";
  check "unsigned" Wasm.unsigned_leb128 624485 "e5 8e 26";
  check "signed" Wasm.signed_leb128 0 "00";
  check "signed" Wasm.signed_leb128 63 "3f";
  check "signed" Wasm.signed_leb128 64 "c0 00";
  check "signed" Wasm.signed_leb128 (-1) "7f";
  check "signed" Wasm.signed_leb128 (-64) "40";
  check "signed" Wasm.signed_leb128 (-65) "bf 7f";
  check "signed" Wasm.signed_leb128 (-123456) "c0 bb 78"

(* The smallest module: header plus empty sections must start with the magic. *)
let () =
  let bytes =
    Wasm.encode { memory_pages = 1; functions = []; constants = []; data = [] }
  in
  if String.sub bytes 0 8 <> "\x00asm\x01\x00\x00\x00" then (
    prerr_endline "FAIL: module header";
    exit 1);
  print_endline "test_wasm: all passed"
