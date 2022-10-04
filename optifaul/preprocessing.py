"""Preprocess data and select most significant features.

Scale time series data from anaerobic digesters and interpolate missing values. Select exogenous (driving) series most
suitable for prediciting the biogas production rate.
"""

from glob import glob
from typing import TYPE_CHECKING, Generator

import pandas as pd

from .utils import date_object_from, get_time_series, new_headers

if TYPE_CHECKING:
    from pandas.core.frame import DataFrame


def _load_raw_data_from(dir_: str) -> Generator["DataFrame", None, None]:
    """Load raw AIZ time series data from Excel files."""
    use_cols = {
        "Faulung": [
            "Datum",
            "Rohs. FB-1 [m³] ",
            "Rohs. FB-2 [m³] ",
            "Rohs. gesamt [m³] ",
            "TS Rohschlamm [g/l] ",  # NaNs
            "Rohs. TS-Fracht [kg/d] ",  # NaNs
            "Rohs. oTS-Fracht [kg/d] ",  # NaNs, correlated
            "Faulschlamm1 Menge [m³] ",
            "Faulschlamm2 Menge [m³] ",
            "Faulschlamm Menge [m³] ",
            "Faulbehälter1 Temperatur [°C] ",
            "Faulbehälter2 Temperatur [°C] ",
            "Faulschlamm1 pH-Wert [-] ",
            "Faulschlamm2 pH-Wert [-] ",
            "Faulbehälter Faulzeit [d] ",
            "TS Faulschlamm [g/l] ",  # NaNs
            "Faulschlamm TS-Fracht [kg/d] ",  # NaNs
            "Faulbehälter Feststoffbelastung [kg/(m³.d)] ",  # NaNs, correlated
            "GV Faulschlamm [%] ",  # NaNs
            "Faulschlamm oTS-Fracht [kg/d] ",  # NaNs, correlated
            "Kofermentation Bioabfälle [m³] ",
            # "Kofermentation CSB-Fracht [kg] ",  # only NaNs
            ],
        "Faulgas": [
            "Faulgas1 Menge [Nm³] ",
            "Faulgas2 Menge [Nm³] ",
            # "CH4 FB-1 [%] ",
            # "CH4 FB-2 [%] ",
        ]
    }
    for type_ in ["Faulung", "Faulgas"]:
        files = glob(f"{dir_}/{type_}-*.xlsx")
        data = pd.concat([pd.read_excel(file_, skiprows=1, skipfooter=6) for file_ in files], ignore_index=True)
        data = data[use_cols[type_]]
        yield data


def _treat_missing_values(data: "DataFrame") -> "DataFrame":
    """Replace missing values in time series data from digester."""
    # data = data.fillna(method="backfill", axis="columns")
    data = data.interpolate(method="linear", axis="columns")
    return data


def _treat_outliers(data: "DataFrame", n_quantiles: int = 4) -> "DataFrame":
    """Find and edit outliers based on quantiles."""
    q_low = data.quantile(1 / n_quantiles)
    q_high = data.quantile((n_quantiles - 1) / n_quantiles)
    iqr = q_high - q_low  # interquartile range
    # Either: Flooring and capping.
    data.where(data > q_low - 1.5 * iqr, q_low, axis=0)
    data.where(data < q_high + 1.5 * iqr, q_high, axis=0)
    # Or: Remove outliers alltogether.
    # data = data[(((data > (q_low - 1.5 * iqr)) & (data < (q_high + 1.5 * iqr)))).all(axis=1)]
    return data


def _prepare_pt_forecasting(data: "DataFrame", dir_: str) -> "DataFrame":
    """Add features, some needed by PyTorch Forecasting library."""
    # Add relative time index and group ids.
    start = data["date"].min()
    data["time_idx"] = (data["date"] - start).dt.days
    data["group_ids"] = 0

    # Add additional time features.
    data["month"] = data["date"].dt.month.astype(str).astype("category")
    data["weekday"] = data["date"].dt.weekday.astype(str).astype("category")

    # Add public holidays, tourism, and ambient temperature.
    data["holidays"] = get_time_series(dir_ + "holidays_tirol.csv", data["date"])["name"]
    # https://www.statistik.at/web_de/statistiken/wirtschaft/tourismus/beherbergung/ankuenfte_naechtigungen/index.html
    data["tourism"] = get_time_series(dir_ + "tourism_strass.csv", data["date"])["overnight_stay"]
    data["tourism"] = data["tourism"].fillna(method="ffill")
    # https://www.wunderground.com/weather/at/strass-im-zillertal
    # data["ambient_temp"] = get_time_series(dir_ + "ambient_temp.csv", data["date"])["Temperatur"]
    return data


def main(config: dict) -> None:
    """Build data loading and preprocessing pipeline."""
    # Load and format data frame.
    digestion, biogas = _load_raw_data_from(config["root_dir"])
    data = digestion.join(biogas)
    data.columns = new_headers()
    dates = date_object_from(data.pop("date"))

    # Treat NaNs and outliers.
    data = _treat_missing_values(data)
    # data = _treat_outliers(data, config["n_quantiles"])

    # Add library-specific features and save to disk.
    data.insert(0, "date", dates)  # CAVE: outliers possibly removed
    data = _prepare_pt_forecasting(data, config["root_dir"])
    data.to_pickle("./assets/data/preprocessed/data.pkl")
