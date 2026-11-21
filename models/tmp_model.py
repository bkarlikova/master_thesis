import os
import torch
import torch.nn as nn
import onnx
from google.protobuf import text_format


# Classic feed forward neural network with one hidden layer, no recurrent connections.
class FeedForwardNN(nn.Module):
    def __init__(self, input_features=10, hidden_units=64, num_classes=2):
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Linear maps the input vector (size `input_features`) to a hidden vector of size hidden_units
            nn.Linear(input_features, hidden_units),
            # Activation: ReLU applies a non-linear function element-wise.
            nn.ReLU(),
            # Layer 2: Linear maps the hidden vector to num_classes outputs -> creates logits.
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# Simple RNN model with one recurrent layer and a linear classifier on top. The number of iteration is specified by the sequence length of the input tensor.
# This will be exported as Add and Mul layers in ONNX


class SimpleRNN(nn.Module):
    def __init__(self, input_features=10, hidden_units=32, num_classes=2):
        super().__init__()
        # Recurrent layer: processes sequences step-by-step.
        # input_size = features per time-step.
        # hidden_size = dimensionality of the hidden state (memory vector).
        # batch_first=True means input tensors are shaped (batch, seq_len, features).
        self.rnn = nn.RNN(
            input_size=input_features, hidden_size=hidden_units, batch_first=True
        )
        # Linear layer to convert the last hidden output to class scores.
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_features)
        # flatten_parameters helps optimize internal weights for some backends.
        self.rnn.flatten_parameters()
        # out: all time-step outputs, shape (batch_size, seq_len, hidden_units)
        # h_n (ignored here) is the final hidden state.
        out, _ = self.rnn(x)
        # Use the last time-step (most recent) output for classification.
        out = out[:, -1, :]
        # Return logits (class scores) for each batch element.
        return self.classifier(out)


# GRU (Gated recurent unit) is a model that is represented always by a single layer of GRU cells. The operation are pretty much the same as for RNN. (flattening, hidden state, taking the last output, returning logits, etc.)
# This will be exported as a single GRU layer in ONNX


class GRUNN(nn.Module):
    def __init__(self, input_features=10, hidden_units=32, num_classes=2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features, hidden_size=hidden_units, batch_first=True
        )
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x):
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.classifier(out)


def export_and_save(
    model, dummy_input, onnx_path, txt_path, input_name="input", output_name="output"
):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={
            input_name: {0: "batch_size"},
            output_name: {0: "batch_size"},
        },
    )

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    pbtxt_path = onnx_path.replace('.onnx', '.pbtxt')
    with open(pbtxt_path, 'w') as ptf:
        ptf.write(text_format.MessageToString(onnx_model))

    with open(txt_path, "w") as f:
        f.write("PyTorch model:\n")
        f.write(str(model))
        f.write("\nONNX graph:\n")
        f.write(onnx.printer.to_text(onnx_model))


def main():
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    batch_size = 1
    seq_len = 8
    features = 10

    ff = FeedForwardNN(input_features=features)
    ff_input = torch.randn(batch_size, features)
    ff_onnx = os.path.join(out_dir, "feedforward.onnx")
    ff_txt = os.path.join(out_dir, "feedforward.txt")
    export_and_save(
        ff, ff_input, ff_onnx, ff_txt, input_name="input_ff", output_name="output_ff"
    )

    rnn = SimpleRNN(input_features=features)
    rnn_input = torch.randn(batch_size, seq_len, features)
    rnn_onnx = os.path.join(out_dir, "rnn.onnx")
    rnn_txt = os.path.join(out_dir, "rnn.txt")
    export_and_save(
        rnn,
        rnn_input,
        rnn_onnx,
        rnn_txt,
        input_name="input_rnn",
        output_name="output_rnn",
    )

    gru = GRUNN(input_features=features)
    gru_input = torch.randn(batch_size, seq_len, features)
    gru_onnx = os.path.join(out_dir, "gru.onnx")
    gru_txt = os.path.join(out_dir, "gru.txt")
    export_and_save(
        gru,
        gru_input,
        gru_onnx,
        gru_txt,
        input_name="input_gru",
        output_name="output_gru",
    )


if __name__ == "__main__":
    main()
