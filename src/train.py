"""Train temporal fusion transformer on anaerobic digester data.

Interpretable multi-horizon time series forecasting with an attention-based neural network architecture. For further
instructions on how to implement the pipeline visit
https://pytorch-forecasting.readthedocs.io/en/latest/tutorials/stallion.html.
"""

from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MASE, RMSE, QuantileLoss
from pytorch_forecasting.models import DecoderMLP, DeepAR, RecurrentNetwork, TemporalFusionTransformer
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


def _create_data_sets(file_: str, split: float, max_encoder_length: int,
                      max_prediction_length: int) -> Tuple["TimeSeriesDataSet", "TimeSeriesDataSet"]:
    """Initialize data sets for PyTorch Forecasting."""
    data = pd.read_pickle(file_)

    train_data = TimeSeriesDataSet(
        data[[col for col in data.columns if "FB1" not in col]],
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
            # "tourism",
            # "ambient_temp",
        ],
    )

    cutoff = int(split * data.shape[0])
    data = data[[col for col in data.columns if "FB2" not in col]]
    val_data = TimeSeriesDataSet(
        data.iloc[cutoff:],
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
    )
    return train_data, val_data


def _tune_hyperparams(train_loader, val_loader) -> None:
    """Search for optimal TFT model hyperparameters."""
    study = optimize_hyperparameters(
        train_loader,
        val_loader,
        model_path="./assets/runs/optuna_test",
        n_trials=64,
        max_epochs=512,
        # gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 32),
        hidden_continuous_size_range=(8, 32),
        attention_head_size_range=(4, 8),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.0, 0.5),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna learning rate finder
    )
    print(f"Best trial params:\n {study.best_trial.params}", flush=True)


def main(config: dict) -> str:
    """Train TFT model on digester time series data."""
    train_data, val_data = _create_data_sets(
        config["train_dir"] + "data.pkl", config["split"], config["max_encoder_length"],
        config["max_prediction_length"])

    # Init dataloaders for model.
    train_loader = train_data.to_dataloader(train=True, batch_size=config["batch_size"], num_workers=4)
    val_loader = val_data.to_dataloader(train=False, batch_size=config["batch_size"], num_workers=4)

    # Hyperparameter study.
    # _tune_hyperparams(train_loader, val_loader)

    # Stop training when loss metric does not improve on validation set.
    lr_logger = LearningRateMonitor()
    # early_stop = EarlyStopping(monitor="val_loss", patience=64, mode="min")
    logger = TensorBoardLogger(config["log_dir"])

    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        gpus=1,
        # gradient_clip_val=config["gradient_clip_val"],
        # limit_train_batches=64,
        # callbacks=[lr_logger, early_stop],
        callbacks=[lr_logger],
        logger=logger,
    )

    # Initialize and train the TFT model.
    model = TemporalFusionTransformer.from_dataset(
        train_data,
        hidden_size=config["hidden_size"],  # biggest influence network size
        lstm_layers=config["lstm_layers"],
        dropout=config["dropout"],
        output_size=7,  # depends on loss function below
        loss=QuantileLoss(),
        attention_head_size=config["attention_head_size"],
        hidden_continuous_size=config["hidden_continuous_size"],
        learning_rate=config["lr"],
        # log_interval=128,
        # reduce_on_plateau_patience=4,  # reduce learning automatically
        # weight_decay=config["weight_decay"],
        # logging_metrics=[],
    )

    # model = DeepAR.from_dataset(
    #         train_data,
    #         allowed_encoder_known_variable_names=["TS Faulschlamm"],
    #         # hidden_size=2,
    #         # rnn_layers=2,
    #         # dropout=0,
    #         logging_metrics=[],
    #         )

    # model = RecurrentNetwork.from_dataset(
    #         train_data,
    #         hidden_size=2,
    #         rnn_layers=2,
    #         # dropout=0,
    #         logging_metrics=[],
    #         )

    # model = DecoderMLP.from_dataset(
    #         train_data,
    #         hidden_size=32,
    #         # n_hidden_layers=4,
    #         dropout=0.0,
    #         output_size=7,
    #         loss=QuantileLoss(),
    #         logging_metrics=[],
    #         )

    trainer.fit(model, train_loader, val_loader)
    with open(f"{logger.log_dir}/hparams.txt", "w") as file_:
        file_.write(str(config))
    return trainer.checkpoint_callback.best_model_path
