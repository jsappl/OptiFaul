"""A collection of helper functions."""

from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from pandas import Series


def new_headers() -> list:
    """Return list of new headers with clean names."""
    return [
        "Datum",
        "Rohs FB1",
        "Rohs FB2",
        "Rohs gesamt",
        "TS Rohschlamm",
        "Rohs TS Fracht",
        # "Rohs oTS Fracht",
        "Faulschlamm Menge FB1",
        "Faulschlamm Menge FB2",
        "Faulschlamm Menge",
        "Temperatur FB1",
        "Temperatur FB2",
        "Faulschlamm pH Wert FB1",
        "Faulschlamm pH Wert FB2",
        "Faulbehaelter Faulzeit",
        "TS Faulschlamm",
        "Faulschlamm TS Fracht",
        # "Faulbehaelter Feststoffbelastung",
        "GV Faulschlamm",
        # "Faulschlamm oTS Fracht",
        "Kofermentation Bioabfaelle",
        # To be predicted.
        "Faulgas Menge FB1",
        "Faulgas Menge FB2",
    ]


def date_object_from(dates: "Series") -> "Series":
    """Convert date strings into datetime objects."""
    return dates.map(lambda x: datetime.strptime(x[:-3], "%d.%m.%Y"))


def get_time_series(file_: str, dates: "Series") -> "Series":
    """Get additional time series data from Austria."""
    data = pd.read_csv(file_, delimiter=";")
    data["Datum"] = data["Datum"].map(lambda x: datetime.strptime(x, "%d.%m.%Y"))
    if isinstance(data.iloc[0, 1], str):
        return pd.merge(dates, data, how="left", on="Datum").fillna("-").astype(str).astype("category")
    return pd.merge(dates, data, how="left", on="Datum")
