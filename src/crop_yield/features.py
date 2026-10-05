"""Construct features available by July 31; no future weather or yield-derived inputs."""

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["COUNTY_ID", "FYEAR"]
WEATHER = ["TMAX", "TMIN", "TAVG", "PREC", "ET0"]


def numeric_features(config: dict) -> list[str]:
    return ["FYEAR", "SM_WHC"] + [
        f"{name}_{month:02d}"
        for month in config["months"]
        for name in ("tavg", "tmax", "tmin", "prec", "et0", "balance")
    ]


def unique_keys(frame: pd.DataFrame, keys: list[str], name: str) -> None:
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f"{name}: missing or duplicate keys {keys}")


def aggregate_weather(meteo: pd.DataFrame, config: dict) -> pd.DataFrame:
    unique_keys(meteo, [*KEY, "DEKAD"], "weather")
    if not meteo.DEKAD.between(1, 36).all() or not (meteo.DEKAD % 1 == 0).all():
        raise ValueError("DEKAD must be an integer from 1 to 36")
    weather = meteo.loc[meteo.DEKAD <= config["cutoff_dekad"]].copy()
    weather["month"] = (weather.DEKAD - 1) // 3 + 1
    weather = weather.loc[weather.month.isin(config["months"])].copy()
    if not np.isfinite(weather[WEATHER].to_numpy()).all():
        raise ValueError("Weather values must be finite")
    if (weather[["PREC", "ET0"]] < 0).any().any():
        raise ValueError("PREC and ET0 must be nonnegative")
    if ((weather.TMIN > weather.TAVG) | (weather.TAVG > weather.TMAX)).any():
        raise ValueError("Temperatures must satisfy TMIN <= TAVG <= TMAX")
    weather["days"] = [
        calendar.monthrange(int(y), int(m))[1] - 20 if int(d) % 3 == 0 else 10
        for y, m, d in zip(weather.FYEAR, weather.month, weather.DEKAD, strict=True)
    ]
    weather["weighted_tavg"] = weather.TAVG * weather.days
    month = weather.groupby([*KEY, "month"], as_index=False).agg(
        tavg=("weighted_tavg", "sum"),
        days=("days", "sum"),
        tmax=("TMAX", "max"),
        tmin=("TMIN", "min"),
        prec=("PREC", "sum"),
        et0=("ET0", "sum"),
        observations=("DEKAD", "count"),
    )
    month["tavg"] /= month.days
    month["balance"] = month.prec - month.et0
    complete_months = month.loc[month.observations == 3]
    counts = complete_months.groupby(KEY).size()
    complete_keys = counts.loc[counts == len(config["months"])].reset_index()[KEY]
    month = month.merge(complete_keys, on=KEY, validate="many_to_one")
    wide = month.pivot(
        index=KEY, columns="month", values=["tavg", "tmax", "tmin", "prec", "et0", "balance"]
    )
    wide.columns = [f"{name}_{int(month):02d}" for name, month in wide.columns]
    return wide.reset_index()


def build_table(raw_dir: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    if config["months"] != [4, 5, 6, 7] or config["cutoff_dekad"] != 21:
        raise ValueError("This benchmark contract is April-July with cutoff dekad 21")
    yields = pd.read_csv(raw_dir / "YIELD_COUNTY_US.csv")
    soil = pd.read_csv(raw_dir / "SOIL_COUNTY_US.csv")
    meteo = pd.read_csv(raw_dir / "METEO_COUNTY_US.csv", usecols=[*KEY, "DEKAD", *WEATHER])
    source_weather_rows = len(meteo)
    unique_keys(yields, KEY, "yield")
    unique_keys(soil, ["COUNTY_ID"], "soil")
    if not np.isfinite(soil.SM_WHC).all() or not (soil.SM_WHC > 0).all():
        raise ValueError("Soil capacity must be finite and positive")
    if not np.isfinite(yields.YIELD).all() or not (yields.YIELD >= 0).all():
        raise ValueError("Yield labels must be finite and nonnegative")
    yields["STATE"] = yields.COUNTY_ID.str.split("_").str[0]
    selected = yields.loc[
        yields.STATE.isin(config["states"])
        & yields.FYEAR.between(config["start_year"], config["test_end"])
    ].copy()
    meteo = meteo.loc[
        meteo.COUNTY_ID.isin(selected.COUNTY_ID)
        & meteo.FYEAR.between(config["start_year"], config["test_end"])
    ].copy()
    weather = aggregate_weather(meteo, config)
    with_soil = selected.merge(
        soil[["COUNTY_ID", "SM_WHC"]], on="COUNTY_ID", validate="many_to_one"
    )
    table = with_soil.merge(weather, on=KEY, validate="one_to_one")
    table = table.sort_values(["FYEAR", "COUNTY_ID"]).reset_index(drop=True)
    if table.empty or not np.isfinite(table[numeric_features(config)].to_numpy()).all():
        raise ValueError("No complete finite feature table")
    audit = {
        "source_yield_rows": len(yields),
        "source_weather_rows": source_weather_rows,
        "source_soil_rows": len(soil),
        "selected_yield_rows": len(selected),
        "source_zero_yield_rows": int((yields.YIELD == 0).sum()),
        "selected_zero_yield_rows": int((selected.YIELD == 0).sum()),
        "retained_zero_yield_rows": int((table.YIELD == 0).sum()),
        "dropped_without_soil": len(selected) - len(with_soil),
        "dropped_without_complete_weather": len(with_soil) - len(table),
        "retained_rows": len(table),
        "counties": int(table.COUNTY_ID.nunique()),
        "states": sorted(table.STATE.unique().tolist()),
        "year_counts": {str(k): int(v) for k, v in table.groupby("FYEAR").size().items()},
        "monthly_dekads_required": 3,
        "cutoff_dekad": 21,
        "target_unit": "bushels/acre",
    }
    return table, audit
