"""Test calibrated temporal fusion transformer on anaerobic digester data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MAPE
from pytorch_forecasting.models import TemporalFusionTransformer


def _create_data_set(
        file_: str, split: float, max_encoder_length: int, max_prediction_length: int) -> "TimeSeriesDataSet":
    """Initialize data sets for PyTorch Forecasting."""
    data = pd.read_pickle(file_)
    cutoff = int(split * data.shape[0])
    data = data[[col for col in data.columns if "FB1" not in col]]
    np.savetxt("./assets/results/actuals.csv", data["Faulgas Menge FB2"].iloc[cutoff:], fmt="%f")
    return TimeSeriesDataSet(
        data.iloc[cutoff:],
        time_idx="time_idx",
        target="Faulgas Menge FB2",
        group_ids=["group_ids"],
        min_encoder_length=0,  # allow predictions without history
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=["month", "weekday", "holidays"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "Rohs FB2",
            "Rohs gesamt",
            "TS Rohschlamm",
            "Rohs TS Fracht",
            # "Rohs oTS Fracht",
            "Faulschlamm Menge FB2",
            "Faulschlamm Menge",
            "Temperatur FB2",
            "Faulschlamm pH Wert FB2",
            "Faulbehaelter Faulzeit",
            "TS Faulschlamm",
            "Faulschlamm TS Fracht",
            # "Faulbehaelter Feststoffbelastung",
            "GV Faulschlamm",
            # "Faulschlamm oTS Fracht",
            "Kofermentation Bioabfaelle",
            "Faulgas Menge FB2",
            "tourism",
            "ambient_temp",
        ],
    )


def main(config: dict) -> None:
    """Test a trained TFT model."""
    best_model_path = "./assets/runs/default/version_6/checkpoints/epoch=23.ckpt"
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    test_data = _create_data_set(
        config["train_dir"] + "data.pkl", config["split"], config["max_encoder_length"],
        config["max_prediction_length"])
    test_loader = test_data.to_dataloader(train=False, batch_size=config["batch_size"], num_workers=4)

    actuals = torch.cat([y for x, y in iter(test_loader)])
    raw_predictions, x = best_tft.predict(test_loader, mode="raw", return_x=True)
    predictions = best_tft.predict(test_loader)

    # Dump 25% and 75% quantiles results to disk.
    np.savetxt(
        "./assets/results/tft.csv", raw_predictions["prediction"][:, 0, [1, 2, 4, 5]], delimiter=",", fmt="%f")

    # Best/worst performers on test data set.
    mean_losses = MAPE(reduction="none")(predictions, actuals).mean(1)
    indices = mean_losses.argsort(descending=True)[12:]  # sort loss and skip NaNs
    for idx in [0, 1, -2, -1]:
        best_tft.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=MAPE())
        plt.savefig(f"./assets/img/best_worst_{idx}.png")

    # Actuals vs predictions by variables.
    # predictions, x = best_tft.predict(test_loader, return_x=True)
    # predictions_vs_actuals = best_tft.calculate_prediction_actual_by_variable(x, predictions)
    # best_tft.plot_prediction_actual_by_variable(predictions_vs_actuals)
    # plt.savefig("./assets/img/variable.png")

    # Importance of different variables.
    interpretation = best_tft.interpret_output(raw_predictions, reduction="sum")
    with open("./assets/results/interpretation.csv", "w") as file_io:
        file_io.write("variable,importance\n")
        total = torch.sum(interpretation["encoder_variables"]).item()
        for idx, imp in enumerate(interpretation["encoder_variables"]):
            file_io.write(f"{best_tft.encoder_variables[idx]},{imp.item() / total * 100}\n")
