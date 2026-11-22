# CLAUDE.md

Master thesis project: compile ONNX neural network models into standalone, model-specialized
WebAssembly modules. The official assignment is in `project_annotation.pdf`.
The user's milestones, plan and scope are in `PROGRESS.md`. Keep that file limited to those topics;
technical notes, open technical questions and the work log belong here.

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
`output_offset`, `output_length`. `runtime/run.mjs` runs it; `models/compare.py` checks against ORT.

## Technical decisions

- Language: OCaml for the compiler (user's choice, confirmed). Python for ONNX Runtime reference
  and model generation; JavaScript for running modules in Node/browser.
- Emit WASM directly, with no C/LLVM toolchain in the pipeline.
- MVP: static shapes, batch size fixed at compile time, weights embedded in the module.

## Open technical questions

- Decided for the "MVP of MVP" (user: as simple as possible): own encoder, batch fixed at compile
  time, one memory region per tensor. Next: MatMul, Add, Sigmoid/Tanh/Softmax (needs own `exp`),
  memory reuse later (proof #4).
- Gemm only supports a bias of shape [N]; general broadcasting is not implemented.
- SIMD128: available in all major browsers; use for matmul after the scalar version works.
  Threads need SharedArrayBuffer + cross-origin isolation, so skip them.
- onnx2c + Emscripten as a baseline: waiting for the user's decision.
- ONNX-MLIR → WASM build is slow and fragile; time-box it, not before February.

## Log

- 2026-11-21: Reviewed annotation and existing code. `dune build` and `onnx-dump` work on
  `feedforward.onnx`. Set up `PROGRESS.md` and this file. User's answers: report = start of the thesis,
  reframing OK, OCaml stays, AI use goes in the university's signed declaration.
- 2026-11-21: Repo moved from `~/FEL/DIP`. First end-to-end run on branch `feature/wasm-codegen-mvp`:
  feedforward.onnx → 3846-byte .wasm, max diff vs ORT 1.8e-7 over 100 inputs (batch 1 and 4).
  The user wants to learn the concepts along the way: explain alternatives, not just code.
