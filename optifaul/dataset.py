"""Load processed data and initialize the data sets."""

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import pandas as pd
from pytorch_forecasting.data.timeseries import TimeSeriesDataSet

if TYPE_CHECKING:
    from pandas import DataFrame


def _load_processed_df(databundle: str = "latest") -> "DataFrame":
    """Load processed databundle into data frame.

    Args:
        databundle: Either choose one or take latest one.
    """
    root = Path.cwd() / "data" / "processed"

    if databundle == "latest":
        data_dir = sorted(root.iterdir())[-1]
    else:
        data_dir = root / databundle

    return pd.read_pickle(data_dir / "samples.pkl")


def init_data_sets(max_encoder_length: int, max_prediction_length: int,
                   databundle: str = "latest") -> Iterator["TimeSeriesDataSet"]:
    """Initialize data sets for PyTorch Forecasting."""
    df = _load_processed_df(databundle)

    for mode in ["train", "val", "test"]:
        if mode == "train":
            df_cut = df[(df.digester == 2) | [x.year not in [2019, 2020] for x in df.date]]
        elif mode == "val":
            df_cut = df[(df.digester == 1) & [x.year == 2020 for x in df.date]]
        elif mode == "test":
            df_cut = df[(df.digester == 1) & [x.year == 2020 for x in df.date]]
        yield TimeSeriesDataSet(
            df_cut,
            time_idx="time_idx",
            target="Biogas D1",
            group_ids=["digester"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_categoricals=["month", "weekday", "public_holiday"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[
                "Raw sludge D1",
                "Biogas D1",
                # "Raw sludge total",
                "Raw sludge dry matter",
                "Raw sludge dry matter load",
                # "Raw sludge organic dry matter load",
                "Sludge D1",
                "Sludge total",
                "Temperature D1",
                "pH value D1",
                "Hydraulic retention time",
                "Sludge dry matter",
                "Sludge dry matter load",
                # "Sediment load",
                "Sludge loss on ignition",
                # "Sludge organic dry matter load",
                "Cofermentation biowaste",
                "overnight_stay",
                "ambient_temp",
            ],
        )
