"""Tests for make_test_models.py. Run from models/ with: uv run python -m unittest"""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import onnx

import make_test_models


class MakeTestModelsTest(unittest.TestCase):
    def test_models_have_dynamic_batch_and_external_weights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with redirect_stdout(StringIO()):
                make_test_models.main(output_dir)

            for name, suffix in (("feedforward", "ff"), ("rnn", "rnn"), ("gru", "gru")):
                with self.subTest(name):
                    path = output_dir / f"{name}.onnx"
                    graph = onnx.load(path).graph
                    self.assertEqual(graph.input[0].name, f"input_{suffix}")
                    self.assertEqual(graph.output[0].name, f"output_{suffix}")
                    for value in (graph.input[0], graph.output[0]):
                        batch = value.type.tensor_type.shape.dim[0]
                        self.assertEqual(batch.dim_param, "batch_size")
                    self.assertGreater((output_dir / f"{name}.onnx.data").stat().st_size, 0)
                    self.assertTrue(path.with_suffix(".txt").exists())
                    self.assertTrue(path.with_suffix(".pbtxt").exists())


if __name__ == "__main__":
    unittest.main()
