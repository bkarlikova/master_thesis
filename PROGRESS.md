# Progress: Compiling ONNX Models into WebAssembly

## Milestones

| Date | Milestone | Status |
| --- | --- | --- |
| 23 Nov 2026 | Supervisor meeting: scope and reframing agreed | |
| 29 Nov 2026 | First end-to-end run: `feedforward.onnx` → `.wasm` → same output as ONNX Runtime | |
| 6 Dec 2026 | MVP complete: all MVP ops, automated comparison with ONNX Runtime | |
| 20 Dec 2026 | Draft of report sections 1–5 | |
| 21 Dec – 4 Jan | *Christmas break* | |
| **17 Jan 2027** | **Research project submitted: MVP + ~10 pages** | |

## Plan

| Week | Dates | Code (Claude) | Me |
| --- | --- | --- | --- |
| 1 | 23–29 Nov | End-to-end slice | Supervisor meeting; review the slice; notes for sections 2–3 |
| 2 | 30 Nov – 6 Dec | Full MVP op set + automated tests | Related work reading (section 4) |
| 3 | 7–13 Dec | First size + cold-start measurement vs ONNX Runtime Web | Softmax proof; error bound sketch; draft sections 2–3 |
| 4 | 14–20 Dec | Fixes from my review | Draft sections 1, 4, 5 |
| — | 21 Dec – 4 Jan | *break* | *break* |
| 5 | 5–11 Jan | Final numbers and figures | Draft section 6; full first version |
| 6 | 12–17 Jan | — | Revise, AI alignment pass, **submit** |

## Scope

### Claim

A model-specialized WASM module beats a general runtime (ONNX Runtime Web) by a large margin
in module size and cold start for small task-specific models, and stays competitive in latency
up to some model size. Finding that crossover point is part of the result.
Correctness is argued from semantics and floating-point error bounds, then checked against ONNX Runtime.

### MVP (by 17 Jan)

- [ ] ONNX model → standalone `.wasm` module, no C/LLVM toolchain
- [ ] Ops: Gemm/MatMul, Add, ReLU, Sigmoid, Tanh, Softmax
- [ ] Runs in Node; output matches ONNX Runtime
- [ ] Automated tests on randomly generated networks
- [ ] *(optional)* first size + cold-start comparison with ONNX Runtime Web

### Not in the MVP

GRU/RNN, convolution, WebGPU, ONNX-MLIR comparison. These come later in the thesis.

### Proofs (my part)

| # | Claim | For January |
| --- | --- | --- |
| 1 | Floating-point error bound, propagated through the network → tolerance for each model | sketch |
| 2 | The generated code computes what the ONNX graph means (semantic preservation) | sketch |
| 3 | Softmax with max-subtraction never overflows and equals the mathematical softmax | full |
| 4 | Memory reuse never overwrites a tensor that is still needed | later |
| 5 | Generated code never reads or writes outside its memory | later |
| 6 | *(stretch)* Machine-checked proof of one part with Why3 | no |

### Report (~10 pages, written by me)

1. Motivation and claim (~1 p)
2. ONNX: graph structure, operators, parsing (~1.5 p)
3. WebAssembly: execution model, memory, SIMD, limits (~1.5 p)
4. Related work: ONNX Runtime Web, ONNX-MLIR, onnx2c, TVM/WebLLM, Burn (~2 p)
5. Compiler design and correctness approach (~2.5 p)
6. MVP status, first results, plan for the thesis (~1.5 p)

### Open decision

- [ ] Include onnx2c + Emscripten as a baseline? (closest existing tool to this idea)
