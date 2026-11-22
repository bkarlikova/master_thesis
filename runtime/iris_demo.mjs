// Classifies the held-out Iris flowers with the compiled WASM module only
// (no Python, no ONNX Runtime).
// Usage: node iris_demo.mjs iris.wasm iris_test.json

import { readFileSync } from "node:fs";

const [wasmPath, testPath] = process.argv.slice(2);
if (!wasmPath || !testPath) {
  console.error("Usage: node iris_demo.mjs iris.wasm iris_test.json");
  process.exit(2);
}

const { instance } = await WebAssembly.instantiate(readFileSync(wasmPath));
const { memory, run, input_offset, input_length, output_offset, output_length } =
  instance.exports;
const { species, features, labels } = JSON.parse(readFileSync(testPath, "utf8"));

const input = new Float32Array(memory.buffer, input_offset.value, input_length.value);
const scores = new Float32Array(memory.buffer, output_offset.value, output_length.value);
const shortName = (index) => species[index].replace("Iris-", "");

let correct = 0;
console.log("sepal len  sepal wid  petal len  petal wid   predicted    actual");
features.forEach((flower, i) => {
  input.set(flower);
  run();
  const predicted = scores.indexOf(Math.max(...scores));
  const ok = predicted === labels[i];
  if (ok) correct++;
  const measurements = flower.map((v) => v.toFixed(1).padStart(9)).join("  ");
  console.log(
    `${measurements}   ${shortName(predicted).padEnd(11)}  ${shortName(labels[i]).padEnd(11)}${ok ? "" : "  <- wrong"}`,
  );
});

console.log(`\n${correct} of ${features.length} flowers classified correctly`);
console.log(`module size: ${readFileSync(wasmPath).length} bytes`);
