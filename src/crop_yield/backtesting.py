"""Expanding-window diagnostics confined to development years, without reselection."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crop_yield.features import numeric_features
from crop_yield.modeling import CountyTrend, metrics, tree_pipeline


def rolling_folds(table: pd.DataFrame, config: dict) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Train on preceding years; predict each development year after ten initial years."""
    start, end = config["start_year"], config["validation_end"]
    first = start + 10
    if end < first or not start <= config["train_end"] < end < config["test_end"]:
        raise ValueError("Require ten initial training years and ordered split boundaries")
    development = table.loc[table.FYEAR.between(start, end)].copy()
    if development.duplicated(["COUNTY_ID", "FYEAR"]).any():
        raise ValueError("Duplicate county-year keys in development data")
    if not np.isfinite(development.YIELD).all():
        raise ValueError("Development targets must be finite")
    if set(development.FYEAR.unique()) != set(range(start, end + 1)):
        raise ValueError("Development data must cover each annual fold")
    folds = []
    for year in range(first, end + 1):
        train = development.loc[development.FYEAR < year].copy()
        validation = development.loc[development.FYEAR == year].copy()
        if len(validation) < 2:
            raise ValueError("Each validation fold needs at least two observations")
        folds.append((train, validation))
    return folds


def candidate_models(config: dict) -> dict:
    """Fixed diagnostic candidates; no search or final-test winner substitution."""
    ridge_preprocess = tree_pipeline(config, "xgboost").named_steps["preprocess"]
    # Scale each training fold after imputation/encoding, never on validation.
    ridge = Pipeline(
        [("preprocess", ridge_preprocess), ("scale", StandardScaler()), ("model", Ridge(alpha=10))]
    )
    return {
        "global_mean": DummyRegressor(strategy="mean"),
        "county_trend": CountyTrend(),
        "ridge": ridge,
        "random_forest": tree_pipeline(config, "random_forest"),
        "xgboost": tree_pipeline(config, "xgboost"),
    }


def run_backtesting(table: pd.DataFrame, config: dict, root: Path) -> dict:
    """Save real predictions and fold evidence, without touching deployed model/test artifacts."""
    folds = rolling_folds(table, config)
    candidates = candidate_models(config)
    scores, predictions = [], []
    for train, validation in folds:
        year = int(validation.FYEAR.iloc[0])
        for name, candidate in candidates.items():
            fitted = clone(candidate).fit(train, train.YIELD)
            pred = fitted.predict(validation)
            scores.append(
                dict(
                    model=name,
                    validation_year=year,
                    train_start=int(train.FYEAR.min()),
                    train_end=int(train.FYEAR.max()),
                    train_rows=len(train),
                    validation_rows=len(validation),
                    **metrics(validation.YIELD, pred),
                )
            )
            frame = validation[["COUNTY_ID", "FYEAR", "YIELD"]].copy()
            frame["model"], frame["prediction"] = name, pred
            predictions.append(frame)
    prediction_table = pd.concat(predictions, ignore_index=True)
    development = table.loc[
        table.FYEAR.between(config["start_year"], config["validation_end"]),
        ["COUNTY_ID", "STATE", "YIELD", *numeric_features(config)],
    ].sort_values(["COUNTY_ID", "FYEAR"])
    result = {
        "protocol": "expanding_window_development_only",
        "development_years": [config["start_year"], config["validation_end"]],
        "excluded_final_test_years": [config["validation_end"] + 1, config["test_end"]],
        "development_sha256": hashlib.sha256(development.to_csv(index=False).encode()).hexdigest(),
        "features": numeric_features(config) + ["STATE"],
        "seed": config["seed"],
        "folds": scores,
        "pooled": {
            name: metrics(group.YIELD, group.prediction)
            for name, group in prediction_table.groupby("model")
        },
        "parameters": {name: repr(model) for name, model in candidates.items()},
        "warning": "Retrospective development diagnostics; test results were already inspected. "
        "No new untouched test, model reselection, causal claim or Moroccan transfer validation.",
    }
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(reports / "development_predictions.csv", index=False)
    (reports / "development_backtest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, group in pd.DataFrame(scores).groupby("model"):
        ax.plot(group.validation_year, group.rmse, marker="o", label=name)
    ax.set(
        xlabel="Development validation year",
        ylabel="RMSE (bushels/acre)",
        title="Past years only: expanding-window development diagnostics",
    )
    ax.legend(ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / "development-backtest.png", dpi=160)
    plt.close(fig)
    return result
