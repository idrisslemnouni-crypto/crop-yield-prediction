"""Expanding-window diagnostics confined to development years, without reselection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crop_yield.features import numeric_features
from crop_yield.modeling import CountyTrend, TrendResidualRegressor, metrics, tree_pipeline


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


def run_backtesting(
    table: pd.DataFrame, config: dict, root: Path, *, residual_only: bool = False
) -> dict:
    """Save real predictions and fold evidence, without touching deployed model/test artifacts."""
    folds = rolling_folds(table, config)
    development = table.loc[
        table.FYEAR.between(config["start_year"], config["validation_end"]),
        ["COUNTY_ID", "STATE", "YIELD", *numeric_features(config)],
    ].sort_values(["COUNTY_ID", "FYEAR"])
    # Final-test NaNs can upcast the whole label column without changing development values.
    development["YIELD"] = development.YIELD.astype(float)
    # Match the historical Windows-produced digest explicitly on every operating system.
    digest = hashlib.sha256(
        development.to_csv(index=False, lineterminator="\r\n").encode()
    ).hexdigest()
    reports = root / "reports"
    reference = None
    if residual_only:
        # Validate saved development predictions before reusing them; never refit old candidates.
        reference = checked_reference(reports, folds, digest, config)
        candidates = {
            "trend_residual_xgboost": TrendResidualRegressor(tree_pipeline(config, "xgboost"))
        }
    else:
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
            frame["YIELD"] = frame.YIELD.astype(float)
            frame["model"], frame["prediction"] = name, pred
            predictions.append(frame)
    prediction_table = pd.concat(predictions, ignore_index=True)
    result = {
        "protocol": "expanding_window_development_only",
        "development_years": [config["start_year"], config["validation_end"]],
        "excluded_final_test_years": [config["validation_end"] + 1, config["test_end"]],
        "development_sha256": digest,
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
    reports.mkdir(parents=True, exist_ok=True)
    stem = "residual" if residual_only else "development"
    plot_scores = scores.copy()
    if reference is not None:
        result["residual_training"] = (
            "in_sample_training_labels_only; trend fitted separately on each past fold"
        )
        result["reused_baselines"] = {
            "predictions_sha256": hashlib.sha256(
                (reports / "development_predictions.csv").read_bytes()
            ).hexdigest(),
            "report_sha256": hashlib.sha256(
                (reports / "development_backtest.json").read_bytes()
            ).hexdigest(),
            "pooled": {
                name: metrics(group.YIELD, group.prediction)
                for name, group in reference.groupby("model")
            },
        }
        for (name, year), group in reference.groupby(["model", "FYEAR"]):
            plot_scores.append(
                dict(model=name, validation_year=year, **metrics(group.YIELD, group.prediction))
            )
    prediction_table.to_csv(reports / f"{stem}_predictions.csv", index=False)
    (reports / f"{stem}_backtest.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    fig = Figure(figsize=(8, 4.5))
    ax = fig.subplots()
    for name, group in pd.DataFrame(plot_scores).groupby("model"):
        ax.plot(group.validation_year, group.rmse, marker="o", label=name)
    ax.set(
        xlabel="Development validation year",
        ylabel="RMSE (bushels/acre)",
        title="Training-only trend + residuals"
        if residual_only
        else "Past years only: expanding-window development diagnostics",
    )
    ax.legend(ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    figures = reports / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / f"{stem}-backtest.png", dpi=160)
    return result


def checked_reference(reports: Path, folds: list, digest: str, config: dict) -> pd.DataFrame:
    """Reject stale/incomplete baseline predictions instead of comparing incompatible runs."""
    report = json.loads((reports / "development_backtest.json").read_text(encoding="utf-8"))
    if report["development_sha256"] != digest:
        raise ValueError("Saved baseline development hash differs; regenerate baselines explicitly")
    if report["seed"] != config["seed"] or report["features"] != numeric_features(config) + [
        "STATE"
    ]:
        raise ValueError("Saved baseline feature/seed protocol differs")
    if report["protocol"] != "expanding_window_development_only":
        raise ValueError("Saved baseline chronology protocol differs")
    reference = pd.read_csv(reports / "development_predictions.csv")
    reference = reference.loc[reference.model.isin(["county_trend", "xgboost"])].copy()
    keys = ["COUNTY_ID", "FYEAR"]
    expected = pd.concat([validation[keys + ["YIELD"]] for _, validation in folds])
    expected = expected.sort_values(keys).reset_index(drop=True)
    for name in ["county_trend", "xgboost"]:
        recorded_folds = [fold for fold in report["folds"] if fold["model"] == name]
        if len(recorded_folds) != len(folds):
            raise ValueError("Saved baseline fold count differs")
        for train, validation in folds:
            year = int(validation.FYEAR.iloc[0])
            recorded = [fold for fold in recorded_folds if fold["validation_year"] == year]
            expected_fold = {
                "train_start": int(train.FYEAR.min()),
                "train_end": int(train.FYEAR.max()),
                "train_rows": len(train),
                "validation_rows": len(validation),
            }
            if len(recorded) != 1 or any(
                recorded[0][key] != value for key, value in expected_fold.items()
            ):
                raise ValueError("Saved baseline training boundaries/counts differ")
        group = reference.loc[reference.model == name].sort_values(keys).reset_index(drop=True)
        if group.duplicated(keys).any() or not group[keys].equals(expected[keys]):
            raise ValueError("Saved baseline county-year keys differ from development folds")
        if (
            not np.allclose(group.YIELD, expected.YIELD, rtol=0, atol=1e-10)
            or not np.isfinite(group.prediction).all()
        ):
            raise ValueError("Saved baseline targets/predictions are invalid")
        for metric, value in metrics(group.YIELD, group.prediction).items():
            if not np.isclose(value, report["pooled"][name][metric], rtol=1e-10, atol=1e-10):
                raise ValueError("Saved baseline metrics do not match saved predictions")
    return reference
