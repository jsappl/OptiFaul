"""Transform part of ETL."""

import datetime as dt
from typing import TYPE_CHECKING, Dict

import pandas as pd

if TYPE_CHECKING:
    from pandas import DataFrame


def keep_translate_cols(dfs: Dict[str, "DataFrame"]) -> Dict[str, "DataFrame"]:
    """Keep only certain columns and translate from German afterwards.

    Args:
        dfs: Dictionary of data frames.

    Returns:
        Dictionary of data frames with selected and translated columns only.
    """
    translate = {
        "digestion":
            {
                "Datum": "date",
                "Rohs. FB-1 [m³] ": "D1 raw sludge",
                "Rohs. FB-2 [m³] ": "D2 raw sludge",
                "Rohs. gesamt [m³] ": "raw sludge total",
                "TS Rohschlamm [g/l] ": "dry matter raw sludge",
                "Rohs. TS-Fracht [kg/d] ": "raw sludge dry matter load",
                "Rohs. oTS-Fracht [kg/d] ": "raw sludge organic dry matter load",
                "Faulschlamm1 Menge [m³] ": "D1 sludge quantity",
                "Faulschlamm2 Menge [m³] ": "D2 sludge quantity",
                "Faulschlamm Menge [m³] ": "sludge quantity",
                "Faulbehälter1 Temperatur [°C] ": "D1 temperature",
                "Faulbehälter2 Temperatur [°C] ": "D2 temperature",
                "Faulschlamm1 pH-Wert [-] ": "D1 pH",
                "Faulschlamm2 pH-Wert [-] ": "D2 pH",
                "Faulbehälter Faulzeit [d] ": "retention time",
                "TS Faulschlamm [g/l] ": "dry matter sludge",
                "Faulschlamm TS-Fracht [kg/d] ": "sludge dry matter load",
                "Faulbehälter Feststoffbelastung [kg/(m³.d)] ": "solids load",
                "GV Faulschlamm [%] ": "glow loss",
                "Faulschlamm oTS-Fracht [kg/d] ": "sludge organic dry matter load",
                "Kofermentation Bioabfälle [m³] ": "cofermentation bio waste",
                # "Kofermentation CSB-Fracht [kg] ": "cofermentation cod load",  # pure NaNs
            },
        "biogas":
            {
                "Datum": "date",
                "Faulgas1 Menge [Nm³] ": "D1 biogas quantity",
                "Faulgas2 Menge [Nm³] ": "D2 biogas quantity",
                # "Faulgas Menge [Nm³] ": "biogas quantity",
                # "Methan Menge [Nm³] ": "methane quantity",
                # "Faulgas zur BARA [Nm³] ": "biogas to BARA",
                # "Schwachgas von BARA [Nm³] ": "lean gas from BARA",
                # "Faulgas CO2-Gehalt [%] ": "",
                # "CO2 FB-1 [%] ": "D1 CO2",
                # "CO2 FB-2 [%] ": "D2 CO2",
                # "Faulgas CH4-Gehalt [%] ": "",
                # "CH4 FB-1 [%] ": "D1 CH4",
                # "CH4 FB-2 [%] ": "D2 CH4",
                # "Faulgas CH4-Gehalt [%] .1": "",
                # "Faulgas H2S-Gehalt [ppm] ": "",
                # "H2S FB-1 [ppm] ": "D1 H2S",
                # "H2S FB-2 [ppm] ": "D2 H2S",
                # "Faulgas H2-Gehalt [%] ": "biogas H2S",
                # "H2 FB-1 [%] ": "D1 H2",
                # "H2 FB-2 [%] ": "D2 H2",
                # "Faulgas H2S-Gehalt [ppm] .1": "",
                # "Faulgas BHKW [Nm³] ": "",
                # "Faulgas GM1-208 [Nm³] ": "",
                # "Faulgas GM2-312 [Nm³] ": "",
                # "Faulgas Heizung [Nm³] ": "",
                # "FB Eisen Dosiermenge [l] ": "iron dosage",
                # "FB Eisen Einkauf [kg] ": "iron purchase",
                # "spez. Faulgasanfall je EW-CSB120 [l/EW/d] ": "",
                # "spez. Faulgasanfall je org. Feststofffracht [l/kg] ": "",
            },
    }

    for name, df in dfs.items():
        if name in ["digestion", "biogas"]:
            lookup = translate[name]
            to_keep = df.loc[:, list(lookup.keys())]
            to_keep.rename(columns=lookup, inplace=True)
            dfs[name] = to_keep

    return dfs


def string_to_datetime(dfs: Dict[str, "DataFrame"]) -> Dict[str, "DataFrame"]:
    """Parse date strings to datetime objects.

    Args:
        dfs: Dictionary of data frames.

    Returns:
        Dictionary of data frames with transformed date columns.
    """
    for name, df in dfs.items():
        if name in ["digestion", "biogas"]:
            df.date = pd.to_datetime(df.date.str[:-3], format="%d.%m.%Y")
        else:
            df.date = pd.to_datetime(df.date, format="%d.%m.%Y")

        dfs[name] = df

    return dfs


def merge(dfs: Dict[str, "DataFrame"]) -> "DataFrame":
    """Merge all data frames into a single one on dates.

    Args:
        dfs: Dictionary of data frames.

    Returns:
        A single merged data frame containing all the data.
    """
    dates = dfs["digestion"].date
    merged = pd.DataFrame(dates, columns=["date"])

    for df in dfs.values():
        merged = merged.merge(df, how="outer", on="date")

    return merged


def sort_by_date(df: "DataFrame") -> "DataFrame":
    """Sort whole data frame by date.

    Args:
        df: Data frame containing all data.

    Returns:
        The same data frame sorted by date.
    """
    df = df.sort_values(by="date")
    df = df.reset_index(drop=True)

    return df


def interpolate_nans(df: "DataFrame") -> "DataFrame":
    """Interpolate some columns in the data frame.

    Args:
        df: Data frame containing all data.

    Returns data frame with interpolated numerical columns.
    """
    to_interpolate = [
        col for col in df.columns if col not in ["date", "public_holiday", "ambient_temp", "overnight_stay"]
    ]
    df.loc[:, to_interpolate] = df.loc[:, to_interpolate].interpolate(
        method="linear", axis="columns", limit_direction="both")
    df.loc[:, "public_holiday"].fillna("---", inplace=True)

    return df
