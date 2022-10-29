"""Transform part of ETL."""

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
                "Rohs. FB-1 [m³] ": "Raw sludge D1",
                "Rohs. FB-2 [m³] ": "Raw sludge D2",
                # "Rohs. gesamt [m³] ": "Raw sludge total",
                "TS Rohschlamm [g/l] ": "Raw sludge dry matter",
                "Rohs. TS-Fracht [kg/d] ": "Raw sludge dry matter load",
                # "Rohs. oTS-Fracht [kg/d] ": "Raw sludge organic dry matter load",
                "Faulschlamm1 Menge [m³] ": "Sludge D1",
                "Faulschlamm2 Menge [m³] ": "Sludge D2",
                "Faulschlamm Menge [m³] ": "Sludge total",
                "Faulbehälter1 Temperatur [°C] ": "Temperature D1",
                "Faulbehälter2 Temperatur [°C] ": "Temperature D2",
                "Faulschlamm1 pH-Wert [-] ": "pH value D1",
                "Faulschlamm2 pH-Wert [-] ": "pH value D2",
                "Faulbehälter Faulzeit [d] ": "Hydraulic retention time",
                "TS Faulschlamm [g/l] ": "Sludge dry matter",
                "Faulschlamm TS-Fracht [kg/d] ": "Sludge dry matter load",
                # "Faulbehälter Feststoffbelastung [kg/(m³.d)] ": "Sediment load",
                "GV Faulschlamm [%] ": "Sludge loss on ignition",
                # "Faulschlamm oTS-Fracht [kg/d] ": "Sludge organic dry matter load",
                "Kofermentation Bioabfälle [m³] ": "Cofermentation biowaste",
                # "Kofermentation CSB-Fracht [kg] ": "Cofermentation COD load",  # pure NaNs
            },
        "biogas":
            {
                "Datum": "date",
                "Faulgas1 Menge [Nm³] ": "Biogas D1",
                "Faulgas2 Menge [Nm³] ": "Biogas D2",
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


def flatten_digesters(df: "DataFrame") -> "DataFrame":
    """Combine data from both digesters in one column."""
    df_d1 = df[[col for col in df.columns if "D2" not in col]]
    df_d1["digester"] = 1

    df_d2 = df[[col for col in df.columns if "D1" not in col]]
    df_d2["digester"] = 2
    df_d2.columns = df_d1.columns
    return pd.concat([df_d1, df_d2], ignore_index=True)


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


def treat_outliers(df: "DataFrame", method: str, n_quantiles: int = 4) -> "DataFrame":
    """Find and edit outliers based on quantiles.

    Args:
        df: Data frame containing all data.
        method: Either cap or remove the outliers.
        n_quantiles: Number of quantiles for outlier computation.

    Returns:
        A data frame with outliers removed.

    Raises:
        NotImplementedError: If the outlier treatment method is not correct.
    """
    sdf = df.select_dtypes(include="number")

    q_low = sdf.quantile(q=1 / n_quantiles, numeric_only=True)
    q_high = sdf.quantile((n_quantiles - 1) / n_quantiles, numeric_only=True)
    iqr = q_high - q_low  # interquartile range

    if method == "cap":
        sdf = sdf.where(cond=sdf > q_low - 1.5 * iqr, other=q_low, axis=0)
        sdf = sdf.where(cond=sdf < q_high + 1.5 * iqr, other=q_high, axis=0)
        df.loc[:, sdf.columns] = sdf
    elif method == "remove":
        to_keep = sdf.index[(((sdf > (q_low - 1.5 * iqr)) & (sdf < (q_high + 1.5 * iqr)))).all(axis=1)]
        df = df.iloc[to_keep]
    else:
        raise NotImplementedError("Choose either 'cap' or 'remove' as an option")

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
    df.loc[:, "overnight_stay"] = df.overnight_stay.fillna(method="ffill").fillna(method="bfill")

    return df


def enrich(df: "DataFrame") -> "DataFrame":
    """Add additional features for forecasting.

    Args:
        df: Data frame containing all data.

    Returns an enriched data frame.
    """
    # relative time index
    start = df.date.min()
    df["time_idx"] = (df.date - start).dt.days

    # additional time features
    df["month"] = df.date.dt.month.astype(str).astype("category")
    df["weekday"] = df.date.dt.weekday.astype(str).astype("category")

    return df
