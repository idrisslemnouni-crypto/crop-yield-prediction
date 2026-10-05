"""Trusted-local artifact loading and schema validation shared by CLI/API."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_artifact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError("Train the model first: python -m crop_yield.cli train")
    # Joblib can execute code: load only the artifact produced by this repository.
    return joblib.load(path)


def validate_rows(artifact: dict, rows: list[dict]) -> pd.DataFrame:
    if not rows or len(rows) > 100:
        raise ValueError("Provide between 1 and 100 rows")
    expected = {"COUNTY_ID", "STATE", *artifact["numeric_features"]}
    for row in rows:
        if set(row) != expected:
            raise ValueError(f"Expected exact fields: {sorted(expected)}")
    frame = pd.DataFrame(rows)
    for name in artifact["numeric_features"]:
        if frame[name].map(lambda x: isinstance(x, (bool, str))).any():
            raise ValueError(f"{name} must be numeric")
        frame[name] = pd.to_numeric(frame[name], errors="raise")
    if not np.isfinite(frame[artifact["numeric_features"]].to_numpy(dtype=float)).all():
        raise ValueError("All numeric fields must be finite")
    config = artifact["config"]
    if (
        not frame.FYEAR.between(config["start_year"], config["test_end"]).all()
        or not (frame.FYEAR % 1 == 0).all()
    ):
        raise ValueError("Year must be an integer in the historical benchmark period 2000–2018")
    if not frame.STATE.isin(config["states"]).all():
        raise ValueError("State outside the eight-state benchmark domain")
    if not frame.COUNTY_ID.isin(artifact["known_counties"]).all():
        raise ValueError("County must have been observed in final training")
    if not (frame.COUNTY_ID.str.split("_").str[0] == frame.STATE).all():
        raise ValueError("County ID and state do not match")
    if not frame.SM_WHC.between(0, 100, inclusive="right").all():
        raise ValueError("Soil capacity must be positive and at most 100")
    for month in config["months"]:
        suffix = f"_{month:02d}"
        low, avg, high = [frame[name + suffix] for name in ["tmin", "tavg", "tmax"]]
        if not ((low >= -60) & (low <= avg) & (avg <= high) & (high <= 60)).all():
            raise ValueError("Temperatures must satisfy -60 <= tmin <= tavg <= tmax <= 60")
        if (frame[["prec" + suffix, "et0" + suffix]] < 0).any().any():
            raise ValueError("Precipitation and ET0 must be nonnegative")
        if not np.allclose(
            frame["balance" + suffix], frame["prec" + suffix] - frame["et0" + suffix], atol=0.001
        ):
            raise ValueError("Monthly water balance must equal precipitation minus ET0")
    return frame


def predict_rows(artifact: dict, rows: list[dict]) -> list[dict]:
    frame = validate_rows(artifact, rows)
    predictions = artifact["model"].predict(frame)
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced non-finite predictions")
    return [
        {
            "county_id": row.COUNTY_ID,
            "year": int(row.FYEAR),
            "yield_bu_ac": float(pred),
            "model": artifact["selected_model"],
            "scope": "Retrospective US county maize benchmark; no Morocco or operational validation",
        }
        for row, pred in zip(frame.itertuples(index=False), predictions, strict=True)
    ]


def predict_file(model_path: Path, input_path: Path) -> list[dict]:
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    return predict_rows(load_artifact(model_path), rows if isinstance(rows, list) else [rows])
