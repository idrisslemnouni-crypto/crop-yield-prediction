"""Chronological validation, fixed candidate models, and immutable final-test evidence."""

import copy
import json
import logging
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from crop_yield.data import FILES, file_hash
from crop_yield.features import numeric_features

LOGGER = logging.getLogger(__name__)


class CountyTrend(RegressorMixin, BaseEstimator):
    """Training-only county intercepts and slopes; global trend for unseen counties."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        frame = X[["COUNTY_ID", "FYEAR"]].copy()
        frame["yield"] = np.asarray(y)
        self.origin_ = float(frame.FYEAR.mean())
        self.global_ = np.polyfit(frame.FYEAR - self.origin_, frame["yield"], 1)
        self.county_ = {}
        for county, group in frame.groupby("COUNTY_ID"):
            if group.FYEAR.nunique() >= 3:
                self.county_[county] = np.polyfit(group.FYEAR - self.origin_, group["yield"], 1)
            else:
                slope = self.global_[0]
                intercept = float((group["yield"] - slope * (group.FYEAR - self.origin_)).mean())
                self.county_[county] = np.array([slope, intercept])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(
            [
                np.polyval(self.county_.get(county, self.global_), year - self.origin_)
                for county, year in zip(X.COUNTY_ID, X.FYEAR, strict=True)
            ]
        )


def metrics(y: pd.Series, pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(root_mean_squared_error(y, pred)),
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
    }


def temporal_split(table: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, ...]:
    boundaries = [config[k] for k in ["start_year", "train_end", "validation_end", "test_end"]]
    if not boundaries[0] <= boundaries[1] < boundaries[2] < boundaries[3]:
        raise ValueError("Split years must be strictly ordered")
    train = table.loc[table.FYEAR.between(boundaries[0], boundaries[1])].copy()
    validation = table.loc[table.FYEAR.between(boundaries[1] + 1, boundaries[2])].copy()
    test = table.loc[table.FYEAR.between(boundaries[2] + 1, boundaries[3])].copy()
    if any(frame.empty for frame in (train, validation, test)):
        raise ValueError("All chronological splits must be non-empty")
    return train, validation, test


def tree_pipeline(config: dict, kind: str, without_weather: bool = False) -> Pipeline:
    columns = ["FYEAR", "SM_WHC"] if without_weather else numeric_features(config)
    preprocessor = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), columns),
            ("state", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["STATE"]),
        ],
        verbose_feature_names_out=False,
    )
    if kind == "random_forest":
        model = RandomForestRegressor(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=4,
            max_features=0.8,
            random_state=config["seed"],
            n_jobs=2,
        )
    elif kind == "xgboost":
        model = XGBRegressor(
            n_estimators=240,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=5,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=config["seed"],
            n_jobs=2,
        )
    else:
        raise ValueError(f"Unknown model {kind}")
    return Pipeline([("preprocess", preprocessor), ("model", model)])


def select_model(train: pd.DataFrame, validation: pd.DataFrame, config: dict) -> tuple[str, dict]:
    candidates = {
        "global_mean": DummyRegressor(strategy="mean"),
        "county_trend": CountyTrend(),
        "random_forest": tree_pipeline(config, "random_forest"),
        "xgboost": tree_pipeline(config, "xgboost"),
    }
    scores = {}
    for name, model in candidates.items():
        LOGGER.info("Fit candidate %s on %s rows", name, len(train))
        model.fit(train, train.YIELD)
        scores[name] = metrics(validation.YIELD, model.predict(validation))
    selected = min(scores, key=lambda name: scores[name]["rmse"])
    return selected, {"models": candidates, "scores": scores}


def year_bootstrap(predictions: pd.DataFrame, config: dict) -> dict:
    groups = {
        year: ((g.YIELD - g.prediction).pow(2).sum(), len(g))
        for year, g in predictions.groupby("FYEAR")
    }
    rng = np.random.default_rng(config["seed"])
    values = []
    years = list(groups)
    for _ in range(config["bootstrap_repeats"]):
        sampled = rng.choice(years, size=len(years), replace=True)
        values.append(
            np.sqrt(sum(groups[y][0] for y in sampled) / sum(groups[y][1] for y in sampled))
        )
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "lower": float(lower),
        "upper": float(upper),
        "unit": "bushels/acre",
        "resampling_unit": "year",
        "years": len(years),
        "warning": "Only three test-year clusters; this interval is fragile and descriptive.",
    }


def run_training(table: pd.DataFrame, config: dict, root: Path, audit: dict) -> dict:
    from crop_yield.reporting import make_figures

    train, validation, test = temporal_split(table, config)
    selected, selection = select_model(train, validation, config)
    ablation = tree_pipeline(config, "xgboost", without_weather=True)
    ablation.fit(train, train.YIELD)
    ablation_metrics = metrics(validation.YIELD, ablation.predict(validation))
    final_train = pd.concat([train, validation], ignore_index=True)
    predictions = test[["COUNTY_ID", "STATE", "FYEAR", "YIELD"]].copy()
    final_models, final_scores = {}, {}
    for name, model in selection["models"].items():
        fitted = copy.deepcopy(model)
        fitted.fit(final_train, final_train.YIELD)
        pred = fitted.predict(test)
        predictions[name] = pred
        final_models[name] = fitted
        final_scores[name] = metrics(test.YIELD, pred)
    predictions["prediction"] = predictions[selected]
    grouped = []
    for dimension in ("FYEAR", "STATE"):
        for value, group in predictions.groupby(dimension):
            grouped.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "n": len(group),
                    **metrics(group.YIELD, group.prediction),
                }
            )
    versions = {
        name: version(name)
        for name in ["numpy", "pandas", "scikit-learn", "xgboost", "shap", "fastapi", "joblib"]
    }
    reports = root / "reports"
    models_dir = root / "models"
    reports.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)
    artifact = {
        "model": final_models[selected],
        "selected_model": selected,
        "config": config,
        "numeric_features": numeric_features(config),
        "versions": versions,
        "known_counties": sorted(final_train.COUNTY_ID.unique().tolist()),
        "training_years": [int(final_train.FYEAR.min()), int(final_train.FYEAR.max())],
    }
    joblib.dump(artifact, models_dir / "model.joblib")
    source_hashes = {name: file_hash(root / "data" / "raw" / name) for name in FILES}
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "versions": versions,
        "config": config,
        "selection_rule": "minimum validation RMSE; never select on final test",
        "selected_model": selected,
        "validation": selection["scores"],
        "test": final_scores,
        "validation_ablation_xgboost_without_weather": ablation_metrics,
        "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
        "data_audit": audit,
        "source_sha256": source_hashes,
        "model_sha256": file_hash(models_dir / "model.joblib"),
        "test_rmse_year_bootstrap_95pct": year_bootstrap(predictions, config),
        "negative_predictions": int((predictions.prediction < 0).sum()),
        "test_counties_unseen_in_final_train": int(
            len(set(test.COUNTY_ID) - set(final_train.COUNTY_ID))
        ),
    }
    predictions.to_csv(reports / "test_predictions.csv", index=False)
    pd.DataFrame(grouped).to_csv(reports / "group_metrics.csv", index=False)
    (reports / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (reports / "data-audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    make_figures(table, predictions, final_models["xgboost"], test, result, reports)
    example = test.iloc[[0]][["COUNTY_ID", "STATE", *numeric_features(config)]].to_dict(
        orient="records"
    )[0]
    (reports / "example-input.json").write_text(json.dumps(example, indent=2), encoding="utf-8")
    LOGGER.info(
        "Selected %s on validation; final-test RMSE %.3f bu/ac",
        selected,
        final_scores[selected]["rmse"],
    )
    return result
