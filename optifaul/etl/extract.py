"""Extract part of ETL."""

from typing import TYPE_CHECKING, Dict

import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

    from pandas import DataFrame


def load_df(data_path: "Path") -> Dict[str, "DataFrame"]:
    """Load raw data from disk.

    Args:
        data_path: Path to directory containing the raw data.

    Returns:
        A list of data frames containing all data.
    """
    dfs = dict()

    for name, german_name in zip(["digestion", "biogas"], ["Faulung", "Faulgas"]):
        files = data_path.glob(f"{german_name}-*.xlsx")
        dfs[name] = pd.concat([pd.read_excel(file_, skiprows=1, skipfooter=6) for file_ in files], ignore_index=True)

    for name in ["public_holiday", "ambient_temp", "tourism_strass"]:
        dfs[name] = pd.read_csv(data_path / f"{name}.csv", delimiter=";")

    return dfs
