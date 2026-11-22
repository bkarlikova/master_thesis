"""Convert trained PyTorch weights (.pt) into a single, self-contained ONNX file.

Usage: uv run python save_model.py <weights.pt> <model.onnx> --architecture feedforward

Save the weights after training with torch.save(model.state_dict(), "model.pt").
Such a file has no architecture in it, so the architecture must be named here.
The layer sizes are read from the shapes of the weights.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import onnx
import onnxruntime  # type: ignore[import-untyped]
import torch
from torch import nn

from architectures import GRUNN, FeedForwardNN, SimpleRNN

ARCHITECTURES = ("feedforward", "rnn", "gru")
OPSET_VERSION = 17
# With an example batch of 1, the exporter hard-codes 1 into the RNN/GRU output shape.
EXAMPLE_BATCH = 2
VERIFIED_BATCHES = (1, 3, 16)
VERIFICATION_TOLERANCE = 1e-5


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        # weights_only prevents a downloaded .pt file from running code when loaded.
        weights = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        raise ValueError(
            f"{path} is not a readable state_dict; "
            "save it with torch.save(model.state_dict(), path)"
        ) from error
    if not isinstance(weights, dict) or not all(
        isinstance(value, torch.Tensor) for value in weights.values()
    ):
        raise ValueError(f"{path} must contain only a state_dict, not a checkpoint")
    return weights


def build_model(
    architecture: str, weights: dict[str, torch.Tensor], sequence_length: int
) -> tuple[nn.Module, torch.Tensor]:
    """Return the model with its weights loaded, and an example input for it."""
    model: nn.Module
    try:
        if architecture == "feedforward":
            hidden, features = weights["net.0.weight"].shape
            classes, _ = weights["net.2.weight"].shape
            model = FeedForwardNN(features, hidden, classes)
            example = torch.zeros(EXAMPLE_BATCH, features)
        elif architecture in ("rnn", "gru"):
            _, features = weights[f"{architecture}.weight_ih_l0"].shape
            classes, hidden = weights["classifier.weight"].shape
            model_class = SimpleRNN if architecture == "rnn" else GRUNN
            model = model_class(features, hidden, classes)
            example = torch.zeros(EXAMPLE_BATCH, sequence_length, features)
        else:
            raise ValueError(f"unknown architecture: {architecture}")
        model.load_state_dict(weights)
    except (KeyError, RuntimeError) as error:
        raise ValueError(
            f"weights do not fit the {architecture} architecture: {error}"
        ) from error
    return model.eval(), example


def export_to_onnx(
    model: nn.Module,
    example: torch.Tensor,
    path: Path,
    *,
    input_name: str = "input",
    output_name: str = "output",
    external_data: bool = False,
) -> None:
    torch.onnx.export(
        model,
        (example,),
        str(path),
        opset_version=OPSET_VERSION,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_shapes=({0: "batch_size"},),
        external_data=external_data,
        verbose=False,
    )
    onnx.checker.check_model(str(path))
    check_opset(path)


def check_opset(path: Path) -> None:
    """The exporter keeps opset 18, without an error, if converting it to 17 fails."""
    imports = onnx.load(path, load_external_data=False).opset_import
    versions = [entry.version for entry in imports if entry.domain in ("", "ai.onnx")]
    if versions != [OPSET_VERSION]:
        raise ValueError(f"{path} has opset {versions}, expected {OPSET_VERSION}")


def verify_export(model: nn.Module, example: torch.Tensor, path: Path) -> None:
    session = onnxruntime.InferenceSession(str(path))
    input_name = session.get_inputs()[0].name
    generator = torch.Generator().manual_seed(0)
    for batch in VERIFIED_BATCHES:
        inputs = torch.randn(batch, *example.shape[1:], generator=generator)
        with torch.no_grad():
            expected = model(inputs).numpy()
        actual = session.run(None, {input_name: inputs.numpy()})[0]
        if not np.allclose(actual, expected, atol=VERIFICATION_TOLERANCE):
            raise ValueError(f"{path} differs from PyTorch at batch size {batch}")


def write_text_dump(model: nn.Module, onnx_path: Path) -> None:
    graph_text = onnx.printer.to_text(onnx.load(onnx_path))
    onnx_path.with_suffix(".txt").write_text(
        f"PyTorch model:\n{model}\nONNX graph:\n{graph_text}"
    )


def convert(
    weights_path: Path,
    onnx_path: Path,
    architecture: str,
    sequence_length: int = 8,
    *,
    input_name: str = "input",
    output_name: str = "output",
) -> nn.Module:
    """Convert and verify. Returns the model as rebuilt from the saved weights."""
    if onnx_path.suffix != ".onnx":
        raise ValueError(f"output must be an .onnx file: {onnx_path}")
    if sequence_length < 1:
        raise ValueError("sequence length must be at least 1")
    model, example = build_model(architecture, load_weights(weights_path), sequence_length)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(
        model, example, onnx_path, input_name=input_name, output_name=output_name
    )
    verify_export(model, example, onnx_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PyTorch weights to ONNX.")
    parser.add_argument("weights_path", type=Path)
    parser.add_argument("onnx_path", type=Path)
    parser.add_argument("--architecture", choices=ARCHITECTURES, required=True)
    parser.add_argument(
        "--sequence-length", type=int, default=8, help="time steps (RNN and GRU only)"
    )
    args = parser.parse_args()

    try:
        convert(args.weights_path, args.onnx_path, args.architecture, args.sequence_length)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(f"Wrote {args.onnx_path}, verified against PyTorch")


if __name__ == "__main__":
    main()
