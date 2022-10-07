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

    params = {
        "train": {
            "skip": "D1",
            "target": "D2",
            "year": [2018, 2019, 2020],
        },
        "val": {
            "skip": "D2",
            "target": "D1",
            "year": [2018, 2019],
        },
        "test": {
            "skip": "D2",
            "target": "D1",
            "year": [2020],
        }
    }

    for mode in ["train", "val", "test"]:
        yield TimeSeriesDataSet(
            df.loc[df.date.dt.year.isin(params[mode]["year"]),
                   [col for col in df.columns if params[mode]["skip"] not in col]],
            time_idx="time_idx",
            target=f"{params[mode]['target']} biogas quantity",
            group_ids=["group_ids"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_categoricals=["month", "weekday", "public_holiday"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[
                f"{params[mode]['target']} raw sludge",
                f"{params[mode]['target']} biogas quantity",
                f"{params[mode]['target']} raw sludge",
                "raw sludge total",
                "dry matter raw sludge",
                "raw sludge dry matter load",
                "raw sludge organic dry matter load",
                f"{params[mode]['target']} sludge quantity",
                "sludge quantity",
                f"{params[mode]['target']} temperature",
                f"{params[mode]['target']} pH",
                "retention time",
                "dry matter sludge",
                "sludge dry matter load",
                "solids load",
                "glow loss",
                "sludge organic dry matter load",
                "cofermentation bio waste",
                "overnight_stay",
                "ambient_temp",
            ],
        )
