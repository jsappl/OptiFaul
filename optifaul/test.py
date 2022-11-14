"""Test calibrated temporal fusion transformer on anaerobic digester data."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_forecasting
import torch

from optifaul.utils import namestr_from, r_squared

if TYPE_CHECKING:
    from pathlib import Path

    from matplotlib.pyplot import Figure
    from torch.nn import Module
    from torch.utils.data import DataLoader


def compute_metrics(model: "Module", loader: "DataLoader", save_dir: "Path") -> None:
    """Evaluate a trained model on various metrics.

    Args:
        model: The model to be evaluated.
        loader: Provides an iterator over the data set.
        save_dir: Where to save the metric results.
    """
    targets = torch.cat([y[0] for _, y in iter(loader)])
    predictions = model.predict(loader)

    file = open(save_dir / "metrics.csv", "a")
    file.write(f"{namestr_from(model)}")

    for name in ["MAE", "MAPE", "SMAPE"]:
        metric = eval(f"pytorch_forecasting.metrics.point.{name}(reduction='none')")
        loss = metric(predictions, targets)
        file.write(f",{round(loss.mean().item(), 2)},{round(loss.std().item(), 2)}")
    metric = eval("pytorch_forecasting.metrics.point.RMSE()")
    file.write(f",{round(metric(predictions, targets).item(), 2)}\n")

    file.close()


def goodness_of_fit(model: "Module", loader: "DataLoader", save_dir: "Path") -> None:
    """Save measurements and forecast to disk.

    Args:
        model: The model to be evaluated.
        loader: Provides an iterator over the data set.
        save_dir: Where to save the metric results.
    """
    targets = torch.cat([y[0] for _, y in iter(loader)]).flatten().numpy()
    predictions = model.predict(loader).flatten().numpy()

    print(f"r2 {namestr_from(model)}: {r_squared(targets, predictions)}")
    np.savetxt(
        save_dir / f"goodness_of_fit_{namestr_from(model)}.csv",
        np.stack((targets, predictions), axis=1),
        fmt="%f",
        delimiter=",",
        header="targets,predictions",
        comments="",
    )


def predictions_plot(model: "Module", loader: "DataLoader") -> "Figure":
    """Plot targets versus predictions of a trained model.

    Args:
        model: The model to be evaluated.
        loader: Provides an iterator over the data set.

    Returns:
        A target vs. predictions plot.
    """
    targets = torch.cat([y[0][:, 1] for _, y in iter(loader)]).flatten().numpy()
    predictions = model.predict(loader)[:, 1].flatten().numpy()
    fig = plt.figure()

    plt.plot(targets, label="targets")
    plt.plot(predictions, label="predictions")

    return fig


def variable_ranking(model: "Module", loader: "DataLoader", save_dir: "Path") -> None:
    """Rank input variables of TFT by importance."""
    file = open(save_dir / "importance.csv", "w")
    file.write("variable,importance\n")

    raw_predictions, _ = model.predict(loader, mode="raw", return_x=True)
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    total = torch.sum(interpretation["encoder_variables"]).item()

    for variable, value in zip(model.encoder_variables, interpretation["encoder_variables"]):
        file.write(f"{variable},{value.item() / total * 100}\n")

    file.close()
