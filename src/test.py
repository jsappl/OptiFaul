"""Test calibrated temporal fusion transformer on anaerobic digester data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import tikzplotlib
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.models import TemporalFusionTransformer


def _create_data_set(
        file_: str, split: float, max_encoder_length: int, max_prediction_length: int) -> "TimeSeriesDataSet":
    """Initialize data sets for PyTorch Forecasting."""
    data = pd.read_pickle(file_)
    cutoff = int(split * data.shape[0])
    data = data[[col for col in data.columns if "FB2" not in col]]
    np.savetxt("./assets/results/actuals.csv", data["Faulgas Menge FB1"].iloc[cutoff:], fmt="%f")
    return TimeSeriesDataSet(
        data.iloc[:cutoff],
        # data.iloc[cutoff:],  # WRONG
        time_idx="time_idx",
        target="Faulgas Menge FB1",
        group_ids=["group_ids"],
        min_encoder_length=0,  # allow predictions without history
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        time_varying_known_categoricals=["month", "weekday", "holidays"],
        time_varying_known_reals=["time_idx"],
        time_varying_unknown_reals=[
            "Rohs FB1",
            "Rohs gesamt",
            "TS Rohschlamm",
            "Rohs TS Fracht",
            # "Rohs oTS Fracht",
            "Faulschlamm Menge FB1",
            "Faulschlamm Menge",
            "Temperatur FB1",
            "Faulschlamm pH Wert FB1",
            "Faulbehaelter Faulzeit",
            "TS Faulschlamm",
            "Faulschlamm TS Fracht",
            # "Faulbehaelter Feststoffbelastung",
            "GV Faulschlamm",
            # "Faulschlamm oTS Fracht",
            "Kofermentation Bioabfaelle",
            "Faulgas Menge FB1",
            # "tourism",
            # "ambient_temp",
        ],
        allow_missings=True,
    )


def main(config: dict, best_model_path: str) -> None:
    """Test a trained TFT model."""
    best_model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)

    test_data = _create_data_set(
        config["train_dir"] + "data.pkl", config["split"], config["max_encoder_length"],
        config["max_prediction_length"])
    test_loader = test_data.to_dataloader(train=False, batch_size=config["batch_size"], num_workers=4)

    actuals = torch.cat([y[0] for x, y in iter(test_loader)])
    predictions = best_model.predict(test_loader)
    raw_predictions, x = best_model.predict(test_loader, mode="raw", return_x=True)
    xx, _ = torch.sort(raw_predictions["prediction"][:, :])
    means = torch.stack((torch.mean(actuals, dim=1), torch.mean(xx[:, :, 3], dim=1)), dim=1)
    np.savetxt("./assets/results/compare_mean.csv", means, delimiter=",", header="actual,predict", fmt="%f")

    # Generate boxplot per day prediction results.
    sns.boxplot(data=abs(actuals - raw_predictions["prediction"][:, :, 3]))
    plt.savefig("/tmp/boxplot.png")

    # Calculate mean absolute error on test set.
    print(f"Mean absolute error {(actuals - torch.nan_to_num(predictions)).abs().mean()}")

    # Dump 25% and 75% quantiles results to disk.
    # np.savetxt(
    #     "./assets/results/tft.csv", raw_predictions["prediction"][:, 0, [1, 2, 4, 5]], delimiter=",", fmt="%f")

    # Worst performers on test data set.
    mean_losses = RMSE(reduction="none")(predictions, actuals).mean(1)
    indices = mean_losses.argsort(descending=True)[12:]  # sort loss and skip NaNs
    for idx in [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]:
        best_model.plot_prediction(x, raw_predictions, idx=indices[idx], add_loss_to_title=RMSE())
        if idx < 0:
            plt.savefig(f"./assets/img/best_{idx}.png")
            tikzplotlib.save(f"./assets/img/best_{idx}.tex")
        else:
            plt.savefig(f"./assets/img/worst_{idx}.png")
            tikzplotlib.save(f"./assets/img/worst_{idx}.tex")
    # for idx in range(len(mean_losses)):
    #     best_model.plot_prediction(x, raw_predictions, idx=idx, add_loss_to_title=RMSE())
    #     plt.savefig(f"./assets/img/test_{idx}.png")

    # Actuals vs predictions by variables.
    # predictions, x = best_model.predict(test_loader, return_x=True)
    # predictions_vs_actuals = best_model.calculate_prediction_actual_by_variable(x, predictions)
    # best_model.plot_prediction_actual_by_variable(predictions_vs_actuals)
    # plt.savefig("./assets/img/variable.png")

    # Importance of different variables.
    interpretation = best_model.interpret_output(raw_predictions, reduction="sum")
    best_model.plot_interpretation(interpretation)
    with open("./assets/results/attention.csv", "w") as file_io:
        file_io.write("variable,importance\n")
        total = torch.sum(interpretation["encoder_variables"]).item()
        for idx, imp in enumerate(interpretation["encoder_variables"]):
            file_io.write(f"{best_model.encoder_variables[idx]},{imp.item() / total * 100}\n")
