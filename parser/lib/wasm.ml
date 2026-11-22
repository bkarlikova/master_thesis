(* A small subset of WebAssembly and its binary encoding.
   Reference: WebAssembly Core Specification, chapter 5 "Binary Format".
   Only what the code generator needs is included. *)

type value_type = I32 | F32

type instr =
  | Local_get of int
  | Local_set of int
  | I32_const of int
  | I32_add
  | I32_mul
  | I32_lt_s
  | F32_const of float
  | F32_load of int  (* static byte offset added to the address on the stack *)
  | F32_store of int
  | F32_add
  | F32_mul
  | F32_max
  | Loop of instr list
  | Br_if of int

(* Every function has type [] -> []; they communicate through memory. *)
type func = { export_name : string; locals : value_type list; body : instr list }

type module_ = {
  memory_pages : int;  (* one page = 64 KiB *)
  functions : func list;
  constants : (string * int) list;  (* exported immutable i32 globals *)
  data : (int * string) list;  (* (offset, bytes) copied into memory at load *)
}

(* LEB128: variable-length integers used everywhere in the format. *)
let rec unsigned_leb128 (buf : Buffer.t) (n : int) : unit =
  if n < 0 then invalid_arg "unsigned_leb128: negative";
  let byte = n land 0x7f and rest = n lsr 7 in
  if rest = 0 then Buffer.add_char buf (Char.chr byte)
  else (
    Buffer.add_char buf (Char.chr (byte lor 0x80));
    unsigned_leb128 buf rest)

let rec signed_leb128 (buf : Buffer.t) (n : int) : unit =
  let byte = n land 0x7f and rest = n asr 7 in
  let sign_bit_set = byte land 0x40 <> 0 in
  if (rest = 0 && not sign_bit_set) || (rest = -1 && sign_bit_set) then
    Buffer.add_char buf (Char.chr byte)
  else (
    Buffer.add_char buf (Char.chr (byte lor 0x80));
    signed_leb128 buf rest)

let add_byte buf byte = Buffer.add_char buf (Char.chr byte)

let add_length_prefixed buf name =
  unsigned_leb128 buf (String.length name);
  Buffer.add_string buf name

let add_vector buf items add_item =
  unsigned_leb128 buf (List.length items);
  List.iter (add_item buf) items

let value_type_code = function I32 -> 0x7f | F32 -> 0x7d

(* Memory access: alignment hint (log2 bytes = 2 for f32), then offset. *)
let add_memarg buf offset =
  unsigned_leb128 buf 2;
  unsigned_leb128 buf offset

let rec add_instr (buf : Buffer.t) (instr : instr) : unit =
  match instr with
  | Local_get i -> add_byte buf 0x20; unsigned_leb128 buf i
  | Local_set i -> add_byte buf 0x21; unsigned_leb128 buf i
  | I32_const n -> add_byte buf 0x41; signed_leb128 buf n
  | I32_add -> add_byte buf 0x6a
  | I32_mul -> add_byte buf 0x6c
  | I32_lt_s -> add_byte buf 0x48
  | F32_const x ->
    add_byte buf 0x43;
    Buffer.add_int32_le buf (Int32.bits_of_float x)
  | F32_load offset -> add_byte buf 0x2a; add_memarg buf offset
  | F32_store offset -> add_byte buf 0x38; add_memarg buf offset
  | F32_add -> add_byte buf 0x92
  | F32_mul -> add_byte buf 0x94
  | F32_max -> add_byte buf 0x97
  | Loop body ->
    add_byte buf 0x03;
    add_byte buf 0x40 (* block type: no result *);
    List.iter (add_instr buf) body;
    add_byte buf 0x0b (* end *)
  | Br_if depth -> add_byte buf 0x0d; unsigned_leb128 buf depth

(* Locals are declared as runs of (count, type). One run per local is valid
   and keeps this simple. *)
let add_function_body buf (func : func) =
  let body = Buffer.create 256 in
  add_vector body func.locals (fun b t ->
      unsigned_leb128 b 1;
      add_byte b (value_type_code t));
  List.iter (add_instr body) func.body;
  add_byte body 0x0b;
  unsigned_leb128 buf (Buffer.length body);
  Buffer.add_buffer buf body

let add_section (out : Buffer.t) (id : int) (fill : Buffer.t -> unit) : unit =
  let contents = Buffer.create 256 in
  fill contents;
  add_byte out id;
  unsigned_leb128 out (Buffer.length contents);
  Buffer.add_buffer out contents

let export_function = 0x00
let export_memory = 0x02
let export_global = 0x03

let encode (m : module_) : string =
  let out = Buffer.create 4096 in
  Buffer.add_string out "\x00asm\x01\x00\x00\x00" (* magic + version 1 *);
  add_section out 1 (fun b ->
      (* type section: the single type [] -> [] *)
      add_vector b [ () ] (fun b () -> add_byte b 0x60; add_byte b 0; add_byte b 0));
  add_section out 3 (fun b ->
      (* function section: every function uses type 0 *)
      add_vector b m.functions (fun b _ -> unsigned_leb128 b 0));
  add_section out 5 (fun b ->
      (* memory section: one memory with a minimum size, no maximum *)
      add_vector b [ m.memory_pages ] (fun b pages ->
          add_byte b 0x00;
          unsigned_leb128 b pages));
  add_section out 6 (fun b ->
      (* global section: immutable i32 constants *)
      add_vector b m.constants (fun b (_, value) ->
          add_byte b (value_type_code I32);
          add_byte b 0x00 (* immutable *);
          add_instr b (I32_const value);
          add_byte b 0x0b));
  add_section out 7 (fun b ->
      (* export section *)
      let functions =
        List.mapi (fun i f -> (f.export_name, export_function, i)) m.functions
      in
      let globals = List.mapi (fun i (name, _) -> (name, export_global, i)) m.constants in
      let memory = [ ("memory", export_memory, 0) ] in
      add_vector b (memory @ functions @ globals) (fun b (name, kind, index) ->
          add_length_prefixed b name;
          add_byte b kind;
          unsigned_leb128 b index));
  add_section out 10 (fun b -> add_vector b m.functions add_function_body);
  add_section out 11 (fun b ->
      (* data section: active segments for memory 0 *)
      add_vector b m.data (fun b (offset, bytes) ->
          add_byte b 0x00;
          add_instr b (I32_const offset);
          add_byte b 0x0b;
          add_length_prefixed b bytes));
  Buffer.contents out
