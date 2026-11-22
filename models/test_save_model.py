"""Tests for save_model.py. Run with: uv run python -m unittest test_save_model"""

import tempfile
import unittest
from pathlib import Path

import onnx
import torch

from architectures import GRUNN, FeedForwardNN, SimpleRNN
from save_model import OPSET_VERSION, check_opset, convert


class ConvertTest(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.weights_path = self.directory / "model.pt"
        self.onnx_path = self.directory / "model.onnx"
        torch.manual_seed(0)

    def test_each_architecture_gives_one_file_with_dynamic_batch(self) -> None:
        models = {
            "feedforward": FeedForwardNN(4, 16, 3),
            "rnn": SimpleRNN(5, 7, 3),
            "gru": GRUNN(5, 7, 3),
        }
        for architecture, model in models.items():
            with self.subTest(architecture):
                torch.save(model.state_dict(), self.weights_path)
                convert(self.weights_path, self.onnx_path, architecture)

                graph = onnx.load(self.onnx_path, load_external_data=False).graph
                for value in (graph.input[0], graph.output[0]):
                    batch = value.type.tensor_type.shape.dim[0]
                    self.assertEqual(batch.dim_param, "batch_size")
                files = sorted(path.name for path in self.directory.iterdir())
                self.assertEqual(files, ["model.onnx", "model.pt"])

    def test_rejects_weights_of_another_architecture(self) -> None:
        torch.save(SimpleRNN().state_dict(), self.weights_path)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            convert(self.weights_path, self.onnx_path, "feedforward")

    def test_rejects_extra_layers(self) -> None:
        weights = FeedForwardNN().state_dict()
        weights["net.4.weight"] = torch.zeros(2, 2)
        torch.save(weights, self.weights_path)
        with self.assertRaisesRegex(ValueError, "do not fit"):
            convert(self.weights_path, self.onnx_path, "feedforward")

    def test_rejects_whole_pickled_model(self) -> None:
        torch.save(FeedForwardNN(), self.weights_path)
        with self.assertRaisesRegex(ValueError, "not a readable state_dict"):
            convert(self.weights_path, self.onnx_path, "feedforward")

    def test_rejects_checkpoint(self) -> None:
        torch.save({"epoch": 3, "model": FeedForwardNN().state_dict()}, self.weights_path)
        with self.assertRaisesRegex(ValueError, "not a checkpoint"):
            convert(self.weights_path, self.onnx_path, "feedforward")

    def test_does_not_overwrite_weights_with_output(self) -> None:
        torch.save(FeedForwardNN().state_dict(), self.weights_path)
        original = self.weights_path.read_bytes()
        with self.assertRaisesRegex(ValueError, "must be an .onnx file"):
            convert(self.weights_path, self.weights_path, "feedforward")
        self.assertEqual(self.weights_path.read_bytes(), original)

    def test_rejects_failed_opset_conversion(self) -> None:
        graph = onnx.helper.make_graph([], "empty", [], [])
        opset = onnx.helper.make_opsetid("", OPSET_VERSION + 1)
        onnx.save(onnx.helper.make_model(graph, opset_imports=[opset]), self.onnx_path)
        with self.assertRaisesRegex(ValueError, "expected 17"):
            check_opset(self.onnx_path)

    def test_rejects_empty_sequence(self) -> None:
        torch.save(SimpleRNN().state_dict(), self.weights_path)
        with self.assertRaisesRegex(ValueError, "at least 1"):
            convert(self.weights_path, self.onnx_path, "rnn", sequence_length=0)


if __name__ == "__main__":
    unittest.main()
