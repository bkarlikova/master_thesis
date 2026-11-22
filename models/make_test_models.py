"""Export untrained test models (feedforward, RNN, GRU) for the compiler.

Usage: uv run python make_test_models.py

Writes <name>.onnx, <name>.onnx.data and readable .txt/.pbtxt dumps to outputs/,
replacing the files that are there.
"""

from pathlib import Path

import onnx
import torch
from google.protobuf import text_format  # type: ignore[import-untyped]

from architectures import GRUNN, FeedForwardNN, SimpleRNN
from save_model import EXAMPLE_BATCH, export_to_onnx, verify_export, write_text_dump

SEED = 0
FEATURES = 10
SEQUENCE_LENGTH = 8


def main(output_dir: Path = Path("outputs")) -> None:
    torch.manual_seed(SEED)
    sequence_shape = (EXAMPLE_BATCH, SEQUENCE_LENGTH, FEATURES)
    test_models = [
        ("feedforward", "ff", FeedForwardNN(FEATURES), (EXAMPLE_BATCH, FEATURES)),
        ("rnn", "rnn", SimpleRNN(FEATURES), sequence_shape),
        ("gru", "gru", GRUNN(FEATURES), sequence_shape),
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, suffix, model, input_shape in test_models:
        path = output_dir / f"{name}.onnx"
        model.eval()
        example = torch.zeros(input_shape)
        # External data on purpose: these models exercise the loader's .onnx.data support.
        export_to_onnx(
            model,
            example,
            path,
            input_name=f"input_{suffix}",
            output_name=f"output_{suffix}",
            external_data=True,
        )
        verify_export(model, example, path)
        write_text_dump(model, path)
        path.with_suffix(".pbtxt").write_text(text_format.MessageToString(onnx.load(path)))
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
