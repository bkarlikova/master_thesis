# Plan: Compiling ONNX Models into WebAssembly

<!-- markdownlint-configure-file {"MD024": {"siblings_only": true}} -->

**Deliverable, 17 Jan 2027:** working compiler + ~10-page report.
Tags show which annotation requirement a task covers:

- **R1** ONNX + parsing
- **R2** WASM + limits
- **R3** compiler (MatMul, Add, activations, Softmax) checked against ONNX Runtime
- **R4** convolution + WebGPU
- **R5** comparison with ORT Web / ONNX-MLIR.

How we know the compiler is right: each week's **Checks** test one more layer, from single
operators up to whole networks, other machines and browsers. `compare.py` only compares the
final output with ORT; the checks add what it can't see.

---

## Week 1 · 23–29 Nov · Finish the required operators

### Claude

- [ ] R3: MatMul, Add (same shape and bias `[N]`), Sigmoid, Tanh, Softmax (last axis, max subtracted)
- [ ] R3: own `exp` in WASM (WASM has no `exp` instruction); test it against `math.exp`
- [ ] R3: test script that generates 200 random networks from these ops, compiles each and compares with ORT
- [ ] R3: Iris model with Softmax at the end, compiled and passing `compare.py`

### Me

- [ ] Decide:
  - Include onnx2c + Emscripten as a baseline?
  - Which browsers to measure?
- [ ] R1 notes:
  - What is inside an `.onnx` file (ModelProto, GraphProto, NodeProto, TensorProto, opset)?
  - Where can weights live (`raw_data`, `float_data`, external `.onnx.data`) and which do we support?
  - Why generated protobuf bindings and not CAISAR or a Python preprocessing step?
  - Which models does the loader refuse, and why refuse instead of guessing

### Checks

- [ ] Each operator alone: one-node models (MatMul, Add, Sigmoid, Tanh, Softmax, Gemm with
  transA/transB, ReLU), odd shapes (1×1, 1×N, 7×3), each compared with ORT
- [ ] Own `exp`: tested on **every** float32 in the range Softmax uses, against float64 `exp`;
  report the largest error (exhaustive, so this counts as a proof)
- [ ] Softmax with extreme inputs (±1e30, all equal, one huge value): no `inf`/`NaN`, outputs sum to 1
- [ ] Random networks: 200 networks; a failing one is saved so it can be rerun
- [ ] Loader rejects what it can't compile, with a clear message, one test each: unknown operator,
  non-float32 types, operator from another domain, batch ≤ 0, more than one named dimension,
  weight file outside the model's folder, missing or short weight file
- [ ] Iris: still 30/30 correct classes after the changes
- [ ] Me: short argument that Softmax never overflows or divides by zero

### Reading

Read:

- [ ] R1: ONNX IR specification, sections on models, graphs, nodes, tensors and external data:
  <https://onnx.ai/onnx/repo-docs/IR.html>
- [ ] R1: Protocol Buffers encoding: how the bytes in `.onnx` become fields (why we generate a decoder):
  <https://protobuf.dev/programming-guides/encoding/>
- [ ] R3: ONNX operator pages for Gemm, MatMul, Add, Sigmoid, Tanh, Softmax, and the broadcasting rules:
  <https://onnx.ai/onnx/operators/>, <https://onnx.ai/onnx/repo-docs/Broadcasting.html>

If time:

- [ ] R1: Girard-Satabin et al., *CAISAR: A platform for Characterizing Artificial Intelligence Safety
  and Robustness*, 2022, arXiv:2206.03044: how it parses ONNX in OCaml (the alternative to ours)
- [ ] R3: Goldberg, *What Every Computer Scientist Should Know About Floating-Point Arithmetic*,
  ACM Computing Surveys 1991, first section (rounding error)
- [ ] R3: Blanchard, D. J. Higham, N. J. Higham, *Accurately Computing the Log-Sum-Exp and Softmax
  Functions*, IMA J. Numer. Anal. 2021, arXiv:1909.03469: why Softmax subtracts the maximum

**Done when:** every op named in R3 compiles and all 200 random networks match ORT.

---

## Week 2 · 30 Nov – 6 Dec · Browser + first comparison

### Claude

- [ ] Iris demo running in a browser page (annotation goal: "directly in the browser")
- [ ] R5: measurement tool, same for every tool compared:
  file size (raw and gzip), fetch, compile, instantiate, first inference,
  median of 1000 inferences, peak memory
- [ ] R5: ORT Web (standard package) measured on Iris in Node and the browser
- [ ] R5: ORT Web reduced-operator build (only Iris's ops); time-box: 2 days

### Me

- [ ] Related work reading. For each tool or paper, answer: does it interpret the model or
  generate code for it? What does it output? Does it run in a browser? What did it measure?
  - From the annotation: ONNX Runtime (Web), ONNX-MLIR (Jin et al.), CAISAR, WASI-NN,
    WebLLM, nn-JIT.web (Jia et al., MobiSys 2024), WeInfer (Chen et al., WWW 2025)
  - Added: TF Lite Micro (MLSys 2021), TinyEngine, onnx2c, ONNC `onnc-wasm` (closest to this work),
    Wang et al., *Anatomizing DL Inference in Web Browsers* (TOSEM 2025)
- [ ] R2 notes. Answer:
  - How is a WASM module structured (sections, types, linear memory, data segments)?
  - Limits: maximum memory, which number types, SIMD width, why threads need cross-origin isolation,
    no direct GPU access
  - Weights are inside the module: what does that mean for file size and compile time?

### Checks

- [ ] Browser vs Node: the same `.wasm` gives bit-identical outputs in the browser page and in Node
- [ ] ORT Web vs native ORT: same outputs within tolerance (so the baseline computes the same thing)
- [ ] Measurement tool: repeated runs give stable numbers (report the spread, not one run)

### Reading

Read:

- [ ] R2: Haas et al., *Bringing the Web up to Speed with WebAssembly*, PLDI 2017: the design paper
- [ ] R2: WebAssembly Core Specification, chapter 2 (structure) and chapter 5 (binary format);
  compare chapter 5 with `parser/lib/wasm.ml`: <https://webassembly.github.io/spec/core/>
- [ ] R5: Ma et al., *Moving Deep Learning into Web Browser: How Far Can We Go?*, WWW 2019,
  arXiv:1901.09388: splits inference into loading, warm-up and inference (our measurement phases)
- [ ] R5: Wang et al., *Anatomizing Deep Learning Inference in Web Browsers*, TOSEM 2025,
  arXiv:2402.05981: the large measurement study (memory overhead, latency gap)
- [ ] R5: ONNX Runtime Web docs and the reduced-operator build:
  <https://onnxruntime.ai/docs/tutorials/web/>, <https://onnxruntime.ai/docs/build/custom.html>

If time:

- [ ] R2: Jangda et al., *Not So Fast: Analyzing the Performance of WebAssembly vs. Native Code*,
  USENIX ATC 2019: where WASM loses speed
- [ ] R2: V8 blog, *Liftoff: a new baseline compiler for WebAssembly*, 2018: why the first run is
  slower than later ones: <https://v8.dev/blog/liftoff>
- [ ] Related work: WASI-NN specification: <https://github.com/WebAssembly/wasi-nn>

**Done when:** one table with all metrics for Iris: this compiler vs ORT Web.

---

## Week 3 · 7–13 Dec · All baselines + accuracy

### Claude

- [ ] R5: onnx2c + Emscripten baseline on the same models
- [ ] R5: ONNX-MLIR → WASM; time-box: 2 days, then record why it failed if it does
- [ ] R5: model-size sweep: networks of width 16–1024 and depth 2–8
- [ ] R3: error of our WASM and of ORT against a float64 reference, per model; set the
  `compare.py` tolerance from it
- [ ] R2: same `.wasm` on ARM (Mac) and x86: are outputs bit-identical? Also ORT Web?

### Me

- [ ] Draft Background: ONNX structure and parsing (R1), WASM and its limits (R2)
- [ ] Draft Design and correctness. Answer:
  - Why generate one module per model instead of interpreting the model at run time?
  - Why our own binary encoder instead of C + Emscripten or LLVM?
  - Why one memory region per tensor, and what are the alternatives?
  - How is correctness checked, and why that tolerance?

### Checks

- [ ] float64 reference: error of our WASM and of ORT against exact-as-possible results;
  the `compare.py` tolerance is set from this, and the report says "largest observed error"
- [ ] Other inputs than random normal: real data (Iris), large values, zeros
- [ ] Batch sizes 1, 3, 16: each row gives the same result as running it alone with batch 1
- [ ] Layer by layer: option to compare every intermediate tensor with ORT, to find which
  layer is wrong when an output fails
- [ ] Every baseline (onnx2c, ONNX-MLIR) also matches ORT, otherwise its timings mean nothing
- [ ] Determinism: ARM vs x86 and across browsers, bit for bit
- [ ] Me: short argument why WASM float results are the same everywhere (from the specification),
  and why tensors can't overwrite each other (each has its own region)

### Reading

Read:

- [ ] R5: Jin et al., *Compiling ONNX Neural Network Models Using MLIR*, 2020, arXiv:2008.08272
- [ ] R5: onnx2c README: <https://github.com/kraiskil/onnx2c>
- [ ] Related work: David et al., *TensorFlow Lite Micro: Embedded Machine Learning on TinyML Systems*,
  MLSys 2021, arXiv:2010.08678: the interpreter-overhead table (interpret vs generate code)
- [ ] R3: Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM 2002, chapter 3
  (rounding error in sums and dot products): background for the error table

If time:

- [ ] Related work: Chen et al., *TVM: An Automated End-to-End Optimizing Compiler for Deep Learning*,
  OSDI 2018, arXiv:1802.04799
- [ ] Related work: Lin et al., *MCUNet: Tiny Deep Learning on IoT Devices* (TinyEngine), NeurIPS 2020,
  arXiv:2007.10319
- [ ] Related work: ONNC `onnc-wasm`: <https://github.com/ONNC/onnc-wasm>
- [ ] Measuring: Georges et al., *Statistically Rigorous Java Performance Evaluation*, OOPSLA 2007:
  how to report times when a JIT compiler is involved

**Done when:** comparison table with every baseline, and the error table.

---

## Week 4 · 14–20 Dec · Convolution and WebGPU (R4)

### Claude

- [ ] R4: Conv (2D, stride and padding, no groups) + whatever else a small CNN needs (e.g. MaxPool, Flatten)
- [ ] R4: small CNN trained on a small image dataset, compiled and matching ORT
- [ ] R4: WebGPU experiment: a matrix multiply in WebGPU vs our WASM, sizes 64–1024,
  time measured including copying data to and from the GPU

### Me

- [ ] R4 notes:
  - What did the compiler need for Conv, and what would a fast version need (im2col, SIMD)?
  - From what matrix size does WebGPU beat WASM, once copying is counted?
  - What would WebGPU as a compiler target need (asynchronous calls, shader generation, browser support)?
- [ ] Draft Problem and questions
- [ ] Draft Related work, grouped: interpret vs generate code / browser inference / compilers
- [ ] Draft Evaluation. Answer:
  - How much smaller and faster to load is the module than ORT Web, and which baseline step explains it?
  - At what model size does ORT Web become faster per inference?
  - How much memory does each use?

### Checks

- [ ] Conv alone: kernel sizes 1 and 3, stride 1 and 2, with and without padding, each vs ORT
- [ ] CNN: outputs match ORT; on test images, same predicted classes as PyTorch
- [ ] WebGPU matmul: results match our WASM within tolerance (GPU float order differs)

### Reading

Read:

- [ ] R4: Dumoulin and Visin, *A guide to convolution arithmetic for deep learning*, 2016,
  arXiv:1603.07285, chapter 2 (shapes, stride, padding)
- [ ] R4: WebGPU Fundamentals, the compute-shader lessons: <https://webgpufundamentals.org/>
- [ ] R4: Jia et al., *Empowering In-Browser Deep Learning Inference on Edge Devices with
  Just-in-Time Kernel Optimizations* (nn-JIT.web), MobiSys 2024, arXiv:2309.08978
- [ ] R4: Chen et al., *WeInfer: Unleashing the Power of WebGPU on LLM Inference in Web Browsers*,
  WWW 2025, doi:10.1145/3696410.3714553
- [ ] R5: Ruan et al., *WebLLM: A High-Performance In-Browser LLM Inference Engine*, 2024,
  arXiv:2412.15803

If time:

- [ ] R4: Chellapilla et al., *High Performance Convolutional Neural Networks for Document Processing*,
  2006: the original im2col idea
- [ ] Evaluation: Mytkowicz et al., *Producing Wrong Data Without Doing Anything Obviously Wrong!*,
  ASPLOS 2009: threats to validity in measurements

**Done when:** CNN matches ORT; WebGPU vs WASM chart exists; all parts drafted except Convolution, WebGPU and thesis plan.

---

## 21 Dec – 4 Jan · Break

---

## Week 5 · 5–11 Jan · Final numbers + full draft

### Claude

- [ ] Rerun every measurement on the final code; figures for the report
- [ ] Fix what I found in review
- [ ] README: how to build, run and reproduce every number

### Me

- [ ] Draft Convolution, WebGPU and thesis plan
- [ ] Full first version; send to the supervisor by 11 Jan

### Checks

- [ ] One command runs every check above on the final code; all pass; the README says how
- [ ] Every number in the report comes from that final run

### Reading

No new reading: check that every claim in the draft cites one of the sources above.

---

## Week 6 · 12–17 Jan · Submit

### Me

- [ ] Supervisor feedback in
- [ ] AI alignment pass (language, consistency)
- [ ] Check every requirement is answered in the report:
  - [ ] R1: ONNX structure and parsing approach
  - [ ] R2: WASM format and its limits
  - [ ] R3: MatMul, Add, ReLU, Sigmoid, Tanh, Softmax compiled; outputs match ORT
  - [ ] R4: Conv prototype and WebGPU experiment
  - [ ] R5: speed, size and load time vs ORT Web and ONNX-MLIR or onnx2c; WebLLM in context
- [ ] **Submit, 17 Jan**
