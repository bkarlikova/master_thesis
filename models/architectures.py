"""The PyTorch networks used as test inputs for the compiler."""

import torch
from torch import nn


# Classic feed forward neural network with one hidden layer, no recurrent connections.
class FeedForwardNN(nn.Module):
    def __init__(
        self, input_features: int = 10, hidden_units: int = 64, num_classes: int = 2
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # Layer 1: Linear maps the input vector (size `input_features`) to a hidden vector of size hidden_units
            nn.Linear(input_features, hidden_units),
            # Activation: ReLU applies a non-linear function element-wise.
            nn.ReLU(),
            # Layer 2: Linear maps the hidden vector to num_classes outputs -> creates logits.
            nn.Linear(hidden_units, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Simple RNN model with one recurrent layer and a linear classifier on top. The number of iteration is specified by the sequence length of the input tensor.
# The export unrolls it: one MatMul + Add + Tanh group per time step.
class SimpleRNN(nn.Module):
    def __init__(
        self, input_features: int = 10, hidden_units: int = 32, num_classes: int = 2
    ) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


# GRU (Gated recurrent unit) model with a single layer of GRU cells. The operations are pretty much the same as for RNN. (flattening, hidden state, taking the last output, returning logits, etc.)
# This will be exported as a single GRU layer in ONNX
class GRUNN(nn.Module):
    def __init__(
        self, input_features: int = 10, hidden_units: int = 32, num_classes: int = 2
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_features, hidden_size=hidden_units, batch_first=True
        )
        self.classifier = nn.Linear(hidden_units, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.gru.flatten_parameters()
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.classifier(out)
