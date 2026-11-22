# CLAUDE.md

Master thesis project: compile ONNX neural network models into standalone, model-specialized
WebAssembly modules. The official assignment is in `project_annotation.pdf` (requirements R1–R5).
`PROGRESS.md` is the user's week-by-week plan to 17 Jan 2027: tasks, checks and reading per week,
tagged R1–R5. Keep it a plan; technical notes, open technical questions and the work log belong here.

## Roles (agreed with the supervisor)

- Claude writes the code.
- The user writes **all thesis/report text**. Claude may only align it (language, consistency,
  feedback). Never draft prose for the report or thesis.
- The user owns design decisions, proofs, evaluation design and code review. Explain design choices
  so they can defend them; keep code reviewable (small commits, readable OCaml).

## Repository

- `parser/`: OCaml (dune). ONNX protobuf bindings generated from `proto/onnx.proto` via
  ocaml-protoc-plugin; `bin/main.ml` is an inspection tool (`onnx-dump`).
- `models/`: Python (uv). `architectures.py` defines the networks. `save_model.py` converts `.pt`
  state_dicts to single-file ONNX and holds the shared export code. `make_test_models.py` exports
  untrained feedforward/RNN/GRU to `models/outputs/` as `.onnx` + external `.onnx.data` + text dumps.
  The committed test models predate its seed: don't regenerate them without asking the user.
  `iris.py` trains the Iris classifier. `compare.py` checks a `.wasm` against native ORT.
- `runtime/`: Node scripts. `run.mjs` runs a module (JSON on stdin/stdout); `iris_demo.mjs`
  checks predicted classes on the 30 Iris test flowers.

## Commands

```sh
cd parser && dune build
cd parser && dune exec onnx-dump -- ../models/outputs/feedforward.onnx
cd models && uv run python make_test_models.py   # overwrites the committed test models
cd models && uv run python save_model.py model.pt model.onnx --architecture feedforward
cd models && uv run python -m unittest           # Python tests
cd parser && dune test
cd parser && dune exec onnx2wasm -- ../models/outputs/feedforward.onnx ../build/feedforward.wasm [batch]
cd models && uv run python compare.py outputs/feedforward.onnx ../build/feedforward.wasm [--batch N]
cd models && uv run python iris.py          # trains Iris → outputs/iris.pt → iris.onnx, plus iris_test.json
node runtime/iris_demo.mjs build/iris.wasm models/outputs/iris_test.json
```

## Compiler pipeline (`parser/lib/`)

`loader.ml` (ONNX → `Ir.graph`, static shapes, external weights) → `memory_plan.ml` (byte offset per
tensor, no reuse) → `codegen.ml` (one exported `run`, loops over linear memory) → `wasm.ml` (own
binary encoder). The module exports `memory`, `run`, and i32 globals `input_offset`, `input_length`,
`output_offset`, `output_length`. It imports nothing. `runtime/run.mjs` runs it.

`compare.py`: 100 standard-normal inputs (seed 0), compares only the final output with native ORT
(`atol = rtol = 1e-5`, not yet justified). It does not see intermediate tensors.

## Technical decisions

- Language: OCaml for the compiler (user's choice, confirmed). Python for ONNX Runtime reference
  and model generation; JavaScript for running modules in Node/browser.
- Emit WASM directly, with no C/LLVM toolchain in the pipeline. Only compiler dependency:
  ocaml-protoc-plugin (+ `protoc`) for decoding ONNX.
- MVP: static shapes, batch size fixed at compile time, weights embedded in the module.
- Correctness is shown by testing (native ORT, float64 reference, per-operator and random-network
  tests), not formal proof: the annotation doesn't require proofs. The user writes short informal
  arguments only (Softmax no overflow, WASM float determinism, disjoint memory regions).
- No relaxed SIMD (results differ across hardware). If SIMD128 is added for matmul, vectorize
  across output columns so results stay bit-identical to the scalar code.
- Threads need SharedArrayBuffer + cross-origin isolation: skip them.

## Open technical questions

- Gemm only supports a bias of shape [N]; general broadcasting is not implemented.
- Known loader bugs (fix with tests in week 1): batch 0 overwrites weights; negative or non-numeric
  batch crashes; int32/double inputs read as float32; custom-domain `Relu` accepted; every symbolic
  dimension becomes the batch; external-data path can point outside the model's folder; missing or
  short weight files raise raw exceptions; RNN/GRU fail with a misleading float32 message.
- Committed `rnn.onnx`/`gru.onnx` still have batch 1 baked into the output shape until regenerated
  (needs the user's OK).
- User decides in week 1: onnx2c + Emscripten as a baseline; browsers (proposed: Chrome, Firefox, Safari).
- ONNX-MLIR → WASM build is slow and fragile: time-boxed to 2 days in week 3; record why if it fails.
- SIMD for matmul: proposed as optional in week 5, not yet confirmed by the user.
- New dependencies need the user's OK first: onnxruntime-web, Emscripten, onnx2c, ONNX-MLIR, plotting.

## Log

- 2026-11-21: Reviewed annotation and existing code. `dune build` and `onnx-dump` work on
  `feedforward.onnx`. Set up `PROGRESS.md` and this file. User's answers: report = start of the thesis,
  reframing OK, OCaml stays, AI use goes in the university's signed declaration.
- 2026-11-21: Repo moved from `~/FEL/DIP`. First end-to-end run on branch `feature/wasm-codegen-mvp`:
  feedforward.onnx → 3846-byte .wasm, max diff vs ORT 1.8e-7 over 100 inputs (batch 1 and 4).
  The user wants to learn the concepts along the way: explain alternatives, not just code.
- 2026-11-22: Iris trained and compiled: 1027-byte .wasm, max diff 1.91e-6, 30/30 test flowers.
  Model scripts committed (`save_model.py`, `iris.py`, `make_test_models.py`); export-warning
  suppression removed as overbuilt (user prefers the simplest fix). `PROGRESS.md` rewritten as a
  week-by-week plan covering all of R1–R5 by 17 Jan, with checks and a reading list per week;
  proofs made optional. Supervisor meeting 23 Nov: ask about the framing around research
  questions and whether a formal proof is expected.
