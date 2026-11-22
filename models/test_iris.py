"""Tests for iris.py. Run from models/ with: uv run python -m unittest"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import numpy_helper

import iris


class IrisTest(unittest.TestCase):
    def test_onnx_contains_the_saved_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with redirect_stdout(StringIO()):
                iris.main(output_dir)

            files = sorted(path.name for path in output_dir.iterdir())
            self.assertEqual(files, ["iris.onnx", "iris.pt", "iris.txt", "iris_test.json"])

            saved = torch.load(output_dir / "iris.pt", weights_only=True)
            graph = onnx.load(output_dir / "iris.onnx").graph
            self.assertEqual(graph.input[0].name, "measurements")
            self.assertEqual(graph.output[0].name, "scores")
            exported = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
            self.assertEqual(exported.keys(), saved.keys())
            for name, values in saved.items():
                np.testing.assert_array_equal(exported[name], values.numpy())


if __name__ == "__main__":
    unittest.main()
