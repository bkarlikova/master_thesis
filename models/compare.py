"""Check a compiled .wasm module against ONNX Runtime on random inputs.

Usage: uv run python compare.py <model.onnx> <model.wasm> [--batch N] [--trials N]
Exits with status 1 if any output differs by more than the tolerance.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnxruntime  # type: ignore[import-untyped]

RUNNER = Path(__file__).resolve().parent.parent / "runtime" / "run.mjs"

# Fixed for now; to be replaced by the per-model bound from the error analysis.
ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5


def run_wasm(wasm_path: Path, inputs: list[np.ndarray]) -> list[np.ndarray]:
    payload = json.dumps([x.ravel().tolist() for x in inputs])
    result = subprocess.run(
        ["node", str(RUNNER), str(wasm_path)],
        input=payload,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.exit(f"WASM runner failed:\n{result.stderr.strip()}")
    return [np.array(y, dtype=np.float32) for y in json.loads(result.stdout)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("wasm_path", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    session = onnxruntime.InferenceSession(str(args.onnx_path))
    (model_input,) = session.get_inputs()
    shape = [d if isinstance(d, int) else args.batch for d in model_input.shape]

    rng = np.random.default_rng(args.seed)
    inputs = [rng.standard_normal(shape).astype(np.float32) for _ in range(args.trials)]
    expected = [session.run(None, {model_input.name: x})[0].ravel() for x in inputs]
    actual = run_wasm(args.wasm_path, inputs)

    max_difference = max(float(np.max(np.abs(e - a))) for e, a in zip(expected, actual))
    failures = sum(
        not np.allclose(a, e, atol=ABSOLUTE_TOLERANCE, rtol=RELATIVE_TOLERANCE)
        for e, a in zip(expected, actual)
    )
    print(f"{args.trials} random inputs, max |onnxruntime - wasm| = {max_difference:.3g}")
    if failures:
        print(f"FAIL: {failures} of {args.trials} outputs outside tolerance")
        return 1
    print("OK: all outputs within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
