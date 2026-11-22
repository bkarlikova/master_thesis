// Runs a module produced by onnx2wasm.
// Usage: node run.mjs model.wasm < inputs.json
// stdin:  a JSON list of inputs, each a flat list of numbers
// stdout: a JSON list of outputs, each a flat list of numbers

import { readFileSync } from "node:fs";

const [wasmPath] = process.argv.slice(2);
if (!wasmPath) {
  console.error("Usage: node run.mjs model.wasm < inputs.json");
  process.exit(2);
}

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath));
const { memory, run, input_offset, input_length, output_offset, output_length } =
  instance.exports;

const inputs = JSON.parse(readFileSync(0, "utf8"));
const outputs = inputs.map((input) => {
  if (input.length !== input_length.value) {
    throw new Error(`expected ${input_length.value} input values, got ${input.length}`);
  }
  new Float32Array(memory.buffer, input_offset.value, input_length.value).set(input);
  run();
  return Array.from(new Float32Array(memory.buffer, output_offset.value, output_length.value));
});

console.log(JSON.stringify(outputs));
