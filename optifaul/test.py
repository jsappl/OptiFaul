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
    """Rank input variables of TFT by importance.

    Args:
        model: The model to be evaluated.
        loader: Provides an iterator over the data set.
        save_dir: Where to save the metric results.
    """
    file = open(save_dir / "importance.csv", "w")
    file.write("variable,importance\n")

    raw_predictions, _ = model.predict(loader, mode="raw", return_x=True)
    interpretation = model.interpret_output(raw_predictions, reduction="sum")
    total = torch.sum(interpretation["encoder_variables"]).item()

    for variable, value in zip(model.encoder_variables, interpretation["encoder_variables"]):
        file.write(f"{variable},{value.item() / total * 100}\n")

    file.close()


def partial_dependencies(model: "Module", loader: "DataLoader", save_dir: "Path") -> None:
    """Analyze the TFT with partial dependency plots.

    Args:
        model: The model to be evaluated.
        loader: Provides an iterator over the data set.
        save_dir: Where to save the metric results.
    """
    data = loader.dataset
    variables = data.time_varying_unknown_reals
    variables.remove("Biogas D1")

    for variable, mean, std in zip(
        [
            "Raw sludge dry matter load",
            "Sludge loss on ignition",
            "Hydraulic retention time",
            "Raw sludge dry matter",
            "Temperature D1",
            "Sludge dry matter load",
            "pH value D1",
        ],
        [
            10375.16,
            59.65,
            28.08,
            55.17,
            38.47,
            5355.92,
            7.53,
        ],
        [
            2843.56,
            2.49,
            6.36,
            8.97,
            0.61,
            1531.13,
            0.41,
        ],
    ):
        file = open(save_dir / f"dependency_{variable.replace(' ', '_')}.csv", "w")
        file.write(f"{variable},mean,std\n")

        for value in np.linspace(mean - 1.5 * std, mean + 1.5 * std, 16):
            data.set_overwrite_values(variable=variable, values=value, target="all")
            output = model.predict(data)
            file.write(f"{value},{output.mean().item()},{output.std().item()}\n")
            data.reset_overwrite_values()

    file.close()


def single_day_forecast(model: "Module", loader: "DataLoader", save_dir: "Path") -> None:
    """Retrieve only one day forecast for plotting.

    Args:
        model: The model to be evaluated.
        loader: Provides an iterator over the data set.
        save_dir: Where to save the metric results.
    """
    targets = torch.cat([y[0] for _, y in iter(loader)]).numpy()[:, 1]
    predictions = model.predict(loader).numpy()[:, 1]

    np.savetxt(
        save_dir / f"single_day_{namestr_from(model)}.csv",
        np.stack((targets, predictions), axis=1),
        fmt="%f",
        delimiter=",",
        header="target,prediction",
        comments="",
    )


def quantiles_single(model: "Module", loader: "DataLoader", save_dir: "Path") -> None:
    """Compute quantiles for a single day forecast.

    Args:
        model: The model to be evaluated.
        loader: Provides an iterator over the data set.
        save_dir: Where to save the metric results.
    """
    quantiles = model.predict(loader, mode="quantiles")
    df = pd.DataFrame(data=quantiles[:, 0, :].numpy(), columns=["q25", "q40", "q50", "q60", "q75"])
    df.to_csv(save_dir / "quantiles_single.csv", index=False)
