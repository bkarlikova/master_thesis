"""Train a small classifier on the Iris dataset and export it to ONNX.

Iris: 150 flowers, 4 measurements each (sepal length/width, petal length/width
in cm), 3 species. Data from the UCI Machine Learning Repository (CC BY 4.0):
https://archive.ics.uci.edu/dataset/53/iris

Writes outputs/iris.onnx and outputs/iris_test.json (the held-out flowers,
used by runtime/iris_demo.mjs to demonstrate inference in WASM).
"""

import csv
import json
import os

import torch
import torch.nn as nn

from tmp_model import FeedForwardNN, export_and_save

SPECIES = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
TEST_SIZE = 30
EPOCHS = 300
LEARNING_RATE = 0.01


def load_iris(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    with open(path) as f:
        rows = list(csv.reader(f))
    features = torch.tensor([[float(v) for v in row[:4]] for row in rows])
    labels = torch.tensor([SPECIES.index(row[4]) for row in rows])
    return features, labels


def accuracy(model: nn.Module, features: torch.Tensor, labels: torch.Tensor) -> float:
    with torch.no_grad():
        predictions = model(features).argmax(dim=1)
    return (predictions == labels).float().mean().item()


def main() -> None:
    torch.manual_seed(0)
    features, labels = load_iris("data/iris.csv")

    # The file is sorted by species, so shuffle before splitting.
    order = torch.randperm(len(labels))
    test_index, train_index = order[:TEST_SIZE], order[TEST_SIZE:]
    train_x, train_y = features[train_index], labels[train_index]
    test_x, test_y = features[test_index], labels[test_index]

    model = FeedForwardNN(input_features=4, hidden_units=16, num_classes=3)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # The whole training set (120 flowers) is one batch, so one step per epoch.
    for epoch in range(1, EPOCHS + 1):
        optimizer.zero_grad()
        loss = loss_function(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(
                f"epoch {epoch:3d}  loss {loss.item():.3f}  "
                f"train accuracy {accuracy(model, train_x, train_y):.0%}"
            )

    print(f"test accuracy {accuracy(model, test_x, test_y):.0%} on {TEST_SIZE} unseen flowers")

    os.makedirs("outputs", exist_ok=True)
    export_and_save(
        model,
        test_x[:1],
        "outputs/iris.onnx",
        "outputs/iris.txt",
        input_name="measurements",
        output_name="scores",
    )
    with open("outputs/iris_test.json", "w") as f:
        json.dump(
            {
                "species": SPECIES,
                "features": test_x.tolist(),
                "labels": test_y.tolist(),
            },
            f,
        )
    print("wrote outputs/iris.onnx and outputs/iris_test.json")


if __name__ == "__main__":
    main()
