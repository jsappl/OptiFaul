"""Benchmark models for comparison.

Models implemented from related papers used as benchmarks for our approach.

Classes:
    KNeighbors: The well-known k-nearest neighbors algorithm.
    MyARIMA: Wrapper for standard ARIMA model.
"""

from typing import TYPE_CHECKING, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .ann import evaluate_ann_on

if TYPE_CHECKING:
    from numpy import ndarray


class KNeighbors:
    """The infamous k-nearest neihbors algorithm."""

    def __init__(self, k_nbors: int, train_exo: "ndarray", train_meth: "ndarray") -> None:
        self.k_nbors = k_nbors
        self.train_exo = train_exo
        self.train_meth = train_meth

    def forward(self, in_: "ndarray") -> "ndarray":
        """Compute average methane based on features of nearest neighbors."""
        out = np.zeros((in_.shape[0], 2))
        for idx in range(in_.shape[0]):
            distances = np.sum((self.train_exo - in_[idx])**2, axis=1)**(1 / 2)
            indices = np.argsort(-distances)[-self.k_nbors:]
            out[idx] = [idx, np.mean(self.train_meth[indices])]
        return out


class MyARIMA:
    """Non-seasonal autoregressive integrated moving average model.

    Attributes:
        p: Order of the autoregressive model.
        d: Degree of differencing.
        q: Order of the moving-average model.
        meth: Methane time series data.
    """

    def __init__(self, p: int, d: int, q: int, train_meth: "ndarray") -> None:
        self.order = (p, d, q)
        self.history = train_meth

    def forward(self, in_: "ndarray") -> "ndarray":
        """Predict biogas time series."""
        out = np.zeros((in_.shape[0], 2))
        from tqdm import tqdm
        for time in tqdm(range(in_.shape[0])):
            model_fit = ARIMA(self.history, order=self.order).fit()
            out[time] = [time + 1, model_fit.forecast()]
            self.history = np.append(self.history, in_[time])
        return out


def _prepare_data(file_: str) -> Tuple["ndarray", "ndarray", "ndarray", "ndarray"]:
    """Load preprocessed data from pickle file."""
    data = pd.read_pickle(file_)
    train_exo = data[[
        "Rohs FB2", "Rohs gesamt", "TS Rohschlamm", "Rohs TS Fracht", "Faulschlamm Menge FB2", "Faulschlamm Menge",
        "Temperatur FB2", "Faulschlamm pH Wert FB2", "Faulbehaelter Faulzeit", "TS Faulschlamm",
        "Faulschlamm TS Fracht", "GV Faulschlamm", "Kofermentation Bioabfaelle", "month", "weekday"
    ]].to_numpy(dtype=float)
    train_meth = data["Faulgas Menge FB2"].to_numpy(dtype=float)
    test_exo = data[[
        "Rohs FB1", "Rohs gesamt", "TS Rohschlamm", "Rohs TS Fracht", "Faulschlamm Menge FB1", "Faulschlamm Menge",
        "Temperatur FB1", "Faulschlamm pH Wert FB1", "Faulbehaelter Faulzeit", "TS Faulschlamm",
        "Faulschlamm TS Fracht", "GV Faulschlamm", "Kofermentation Bioabfaelle", "month", "weekday"
    ]].to_numpy(dtype=float)
    test_meth = data["Faulgas Menge FB1"].to_numpy(dtype=float)
    return train_exo, train_meth, test_exo, test_meth


def main(config: dict) -> None:
    """Evaluate all benchmark models."""
    train_exo, train_meth, test_exo, test_meth = _prepare_data(config["train_dir"] + "data.pkl")
    cutoff = 0
    test_exo = test_exo[cutoff:]
    test_meth = test_meth[cutoff:]

    np.savetxt(
        "./assets/results/true.csv", np.stack((list(range(len(test_meth))), test_meth), axis=1), fmt="%f",
        delimiter=",", header="time,pred", comments="")

    knn = KNeighbors(8, train_exo, train_meth)
    out_knn = knn.forward(test_exo)
    np.savetxt("./assets/results/knn.csv", out_knn, fmt="%f", delimiter=",", header="time,pred", comments="")

    # p: number of lag observations
    # d: number of times that the raw observations are differenced
    # q: size of the moving average window
    arima = MyARIMA(8, 1, 0, np.diff(train_meth))
    out_arima = arima.forward(np.diff(test_meth))
    out_arima = np.append(np.array([[0, test_meth[0]]]), out_arima, axis=0)
    out_arima[:, -1] = np.cumsum(out_arima[:, -1])
    # Without differencing.
    # arima = MyARIMA(3, 1, 1, train_meth)
    # out_arima = arima.forward(test_meth)
    np.savetxt("./assets/results/arima.csv", out_arima, fmt="%f", delimiter=",", header="time,pred", comments="")

    preds = evaluate_ann_on(train_exo, train_meth, test_exo)
    np.savetxt("./assets/results/ann.csv", preds, fmt="%f", delimiter=",", header="time,pred", comments="")
