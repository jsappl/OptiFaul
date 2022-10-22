"""Test calibrated temporal fusion transformer on anaerobic digester data."""

from typing import TYPE_CHECKING

import numpy as np
import pytorch_forecasting
import torch

from optifaul.utils import namestr_from, r_squared

if TYPE_CHECKING:
    from pathlib import Path

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

    for name in ["MAE", "MAPE", "RMSE", "SMAPE"]:
        metric = eval(f"pytorch_forecasting.metrics.point.{name}(reduction='none')")
        loss = metric(predictions, targets)
        file.write(f",{round(loss.mean().item(), 2)},{round(loss.std().item(), 2)}")

    file.write("\n")
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


