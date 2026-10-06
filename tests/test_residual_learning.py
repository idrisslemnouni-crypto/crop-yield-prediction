"""Leakage and evidence preservation checks with artificial, cheap fixtures only."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyRegressor

from crop_yield import backtesting
from crop_yield.features import numeric_features
from crop_yield.modeling import CountyTrend, TrendResidualRegressor, tree_pipeline


def fixture_table(config):
    # Synthetic dataset created for demonstration purposes (unit-test fixture only).
    return pd.DataFrame(
        [
            dict.fromkeys(numeric_features(config), 1.0)
            | {
                "FYEAR": year,
                "COUNTY_ID": county,
                "STATE": "IA",
                "YIELD": 2 * (year - 2000) + intercept,
            }
            for year in range(2000, 2019)
            for county, intercept in [("IA_A", 100), ("IA_B", 140)]
        ]
    )


def test_perfect_trend_extrapolates_and_estimator_is_cloneable(config):
    table = fixture_table(config)
    train = table.loc[table.FYEAR < 2010]
    prototype = TrendResidualRegressor(DummyRegressor())
    fitted = clone(prototype).fit(train, train.YIELD)
    future = table.loc[table.FYEAR == 2015].copy()
    truth = future.YIELD.copy()
    future["YIELD"] = -1e9  # Label carried in the frame must not enter predictions.
    assert np.allclose(fitted.predict(future), truth)
    assert not hasattr(prototype, "trend_")
    assert fitted.residual_estimator_ is not prototype.residual_estimator
    assert abs(fitted.residual_estimator_.constant_[0, 0]) < 1e-10
    pd.testing.assert_frame_equal(train, table.loc[table.FYEAR < 2010])


def test_preprocessing_fits_training_only_and_refit_discards_old_counties(config):
    table = fixture_table(config)
    train = table.loc[table.FYEAR < 2010].copy()
    train["SM_WHC"] = np.tile([np.nan, 2, 4, 6], 5)
    model = TrendResidualRegressor(tree_pipeline(config, "xgboost")).fit(train, train.YIELD)
    imputer = model.residual_estimator_.named_steps["preprocess"].named_transformers_["numeric"]
    assert imputer.statistics_[numeric_features(config).index("SM_WHC")] == 4
    before = imputer.statistics_.copy()
    future = table.loc[table.FYEAR == 2015].copy()
    future["SM_WHC"] = 1e12
    model.predict(future)
    np.testing.assert_array_equal(imputer.statistics_, before)
    restricted = train.loc[train.COUNTY_ID == "IA_A"]
    model.fit(restricted, restricted.YIELD)
    assert set(model.trend_.county_) == {"IA_A"}
    assert np.isfinite(model.predict(future)).all()  # Unseen county uses training global trend.


@pytest.fixture
def baseline(config, tmp_path, monkeypatch):
    monkeypatch.setattr(
        backtesting,
        "candidate_models",
        lambda _: {"county_trend": CountyTrend(), "xgboost": DummyRegressor()},
    )
    backtesting.run_backtesting(fixture_table(config), config, tmp_path)
    # Avoid six real tree trainings in orchestration tests.
    monkeypatch.setattr(backtesting, "tree_pipeline", lambda *_: DummyRegressor())
    return tmp_path


def test_residual_runner_isolated_from_final_test_and_preserves_baselines(config, baseline):
    reports = baseline / "reports"
    original = {p: p.read_bytes() for p in reports.rglob("*") if p.is_file()}
    frame = fixture_table(config)
    first = backtesting.run_backtesting(frame, config, baseline, residual_only=True)
    predictions = (reports / "residual_predictions.csv").read_bytes()
    frame.loc[frame.FYEAR >= 2016, "YIELD"] = np.nan
    frame.loc[frame.FYEAR >= 2016, "SM_WHC"] = 1e12
    second = backtesting.run_backtesting(frame, config, baseline, residual_only=True)
    assert first == second
    assert predictions == (reports / "residual_predictions.csv").read_bytes()
    assert all(p.read_bytes() == content for p, content in original.items())
    assert len(first["folds"]) == 6
    assert not (baseline / "models").exists()
    assert not (reports / "test_predictions.csv").exists()


@pytest.mark.parametrize(
    "corruption",
    ["hash", "seed", "keys", "targets", "predictions", "scores", "protocol", "chronology"],
)
def test_incompatible_cached_baselines_rejected_before_fit(
    config, baseline, monkeypatch, corruption
):
    reports = baseline / "reports"
    path = reports / "development_backtest.json"
    saved = json.loads(path.read_text())
    if corruption == "hash":
        saved["development_sha256"] = "invalid"
    elif corruption == "seed":
        saved["seed"] = -1
    elif corruption == "scores":
        saved["pooled"]["xgboost"]["rmse"] += 10
    elif corruption == "protocol":
        saved["protocol"] = "random_split_all_years"
    elif corruption == "chronology":
        saved["folds"][0]["train_end"] = 2018
    else:
        csv = reports / "development_predictions.csv"
        frame = pd.read_csv(csv)
        if corruption == "keys":
            frame = frame.iloc[1:]
        elif corruption == "targets":
            frame.loc[0, "YIELD"] += 100
        else:
            frame.loc[0, "prediction"] = np.nan
        frame.to_csv(csv, index=False)
    path.write_text(json.dumps(saved))

    def unexpected_fit(*_):
        raise AssertionError("Training must not start with an incompatible comparison")

    monkeypatch.setattr(TrendResidualRegressor, "fit", unexpected_fit)
    with pytest.raises(ValueError):
        backtesting.run_backtesting(fixture_table(config), config, baseline, residual_only=True)
