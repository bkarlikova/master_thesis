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
- `models/`: Python (uv). `tmp_model.py` exports test models (feedforward, RNN, GRU) to
  `models/outputs/` as `.onnx` + external `.onnx.data` + text dumps.

## Commands

```sh
cd parser && dune build
cd parser && dune exec onnx-dump -- ../models/outputs/feedforward.onnx
cd models && uv run python tmp_model.py
```

## Technical decisions

- Language: OCaml for the compiler (user's choice, confirmed). Python for ONNX Runtime reference
  and model generation; JavaScript for running modules in Node/browser.
- Emit WASM directly, with no C/LLVM toolchain in the pipeline.
- MVP: static shapes, batch size fixed at compile time, weights embedded in the module.

## Open technical questions

- WASM emission: the `wasm` opam package (WebAssembly reference interpreter, has an AST + binary
  encoder + validator) vs. our own small binary encoder (~300 lines). Adding a dependency needs the
  user's approval.
- External weights: models use `.onnx.data` (external data). The parser must load it.
- Feedforward exports `Gemm` (with `transB=1`), not MatMul+Add. Support both.
- Batch dimension is symbolic (`batch_size`). Decide: fix at compile time, or loop over batch at runtime.
- SIMD128: available in all major browsers; use for matmul after the scalar version works.
  Threads need SharedArrayBuffer + cross-origin isolation, so skip them.
- onnx2c + Emscripten as a baseline: waiting for the user's decision.
- ONNX-MLIR → WASM build is slow and fragile; time-box it, not before February.

## Housekeeping

- `parser/_build/` is committed to git; add it to `.gitignore` and remove it from the index (ask first).

## Log

- 2026-11-21: Reviewed annotation and existing code. `dune build` and `onnx-dump` work on
  `feedforward.onnx`. Set up `PROGRESS.md` and this file. User's answers: report = start of the thesis,
  reframing OK, OCaml stays, AI use goes in the university's signed declaration.
