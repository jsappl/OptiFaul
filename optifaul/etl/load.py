"""Load part of ETL."""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame


def to_disk(df: "DataFrame") -> None:
    """Save preprocessed data frame to disk.

    Args:
        df: Data frame containing all data.
    """
    data_dir = Path.cwd() / "data" / "processed" / f"databundle_{str(datetime.now().date())}"
    data_dir.mkdir(parents=True, exist_ok=False)

    df.to_pickle(data_dir / "samples.pkl")
