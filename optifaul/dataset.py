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
            target=f"Biogas {params[mode]['target']}",
            group_ids=["group_ids"],
            max_encoder_length=max_encoder_length,
            max_prediction_length=max_prediction_length,
            time_varying_known_categoricals=["month", "weekday", "public_holiday"],
            time_varying_known_reals=["time_idx"],
            time_varying_unknown_reals=[
                f"Raw sludge {params[mode]['target']}",
                f"Biogas {params[mode]['target']}",
                # "Raw sludge total",
                "Raw sludge dry matter",
                "Raw sludge dry matter load",
                # "Raw sludge organic dry matter load",
                f"Sludge {params[mode]['target']}",
                "Sludge total",
                f"Temperature {params[mode]['target']}",
                f"pH value {params[mode]['target']}",
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
