"""Train a benchmark ANN.

Classes:
    BiogasData: Time series data of fermentation in digester.
    QdaisANN2010: Artificial neural network inspired by the publication.
"""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from numpy import ndarray
    from torch import Tensor


class BiogasData(Dataset):
    """Digester biogas time series data set."""

    def __init__(self, train_exo: "ndarray", train_meth: "ndarray") -> None:
        self.train_exo = _to_tensor(train_exo)
        self.train_meth = _to_tensor(train_meth).unsqueeze(-1)

    def __len__(self):
        return self.train_exo.shape[0]

    def __getitem__(self, idx):
        return self.train_exo[idx], self.train_meth[idx]


class QdaisANN2010(nn.Module):
    """Simple ANN benchmark model.

    An ANN with two hidden layers, cf. https://www.sciencedirect.com/science/article/pii/S092134490900202X.
    """

    def __init__(self, n_input: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_input, 25),
            nn.ReLU(),
            nn.Linear(25, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, in_: "Tensor") -> "Tensor":
        """Forward pass through network."""
        return self.layers(in_)


def _to_tensor(array: "ndarray") -> "Tensor":
    """Convert numpy array to PyTorch tensor."""
    return torch.from_numpy(array).float()


def _get_trained_ann(train_exo: "ndarray", train_meth: "ndarray") -> "QdaisANN2010":
    """Return trained ANN."""
    train_data = BiogasData(train_exo, train_meth)
    ann = QdaisANN2010(train_data.train_exo.shape[1])
    try:
        ann.load_state_dict(torch.load("./assets/ann.pt"))
    except IOError:
        optimizer = optim.Adam(ann.parameters())
        scheduler = ReduceLROnPlateau(optimizer)
        criterion = nn.MSELoss()
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        for _ in range(1024):  # number of epochs
            running_loss = 0
            for _, (inputs, labels) in enumerate(train_loader):
                optimizer.zero_grad()
                loss = criterion(ann(inputs), labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            scheduler.step(running_loss)
            print(f"Running loss: {running_loss / len(train_data)}", end="\r", flush=True)
        torch.save(ann.state_dict(), "./assets/ann.pt")
    return ann


def evaluate_ann_on(train_exo: "ndarray", train_meth: "ndarray", test_exo: "ndarray") -> "ndarray":
    """Evalute trained ANN on test data."""
    ann = _get_trained_ann(train_exo, train_meth)
    test_exo = torch.from_numpy(test_exo).float()
    n_test = test_exo.shape[0]
    preds = torch.zeros((n_test, 2))
    for idx in range(n_test):
        inputs = test_exo[idx]
        preds[idx] = torch.tensor([idx, ann(inputs)])
    return preds.numpy()
