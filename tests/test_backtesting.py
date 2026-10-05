"""Chronology and holdout isolation contracts, using artificial fixtures only."""

import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor

from crop_yield import backtesting
from crop_yield.features import numeric_features


def development_frame(config):
    rows = []
    for year in range(2000, 2019):
        for county in ("IA_A", "IA_B"):
            rows.append(
                dict.fromkeys(numeric_features(config), 1.0)
                | {"COUNTY_ID": county, "STATE": "IA", "FYEAR": year, "YIELD": year - 1900.0}
            )
    return pd.DataFrame(rows)


def test_folds_use_only_past_and_never_final_test(config):
    folds = backtesting.rolling_folds(development_frame(config), config)
    assert len(folds) == 6
    for train, validation in folds:
        assert train.FYEAR.max() < validation.FYEAR.min() <= 2015
        assert set(train.index).isdisjoint(validation.index)
        assert train.FYEAR.min() == 2000


def test_final_test_poisoning_cannot_change_any_diagnostic(config, tmp_path, monkeypatch):
    monkeypatch.setattr(backtesting, "candidate_models", lambda _: {"mean": DummyRegressor()})
    frame = development_frame(config)
    first = backtesting.run_backtesting(frame, config, tmp_path)
    original_predictions = (tmp_path / "reports/development_predictions.csv").read_bytes()
    frame.loc[frame.FYEAR >= 2016, "YIELD"] = np.nan
    frame.loc[frame.FYEAR >= 2016, "SM_WHC"] = 1e12
    second = backtesting.run_backtesting(frame, config, tmp_path)
    assert first == second
    assert (tmp_path / "reports/development_predictions.csv").read_bytes() == original_predictions
    assert not (tmp_path / "models").exists()
    assert not (tmp_path / "reports/test_predictions.csv").exists()


def test_future_development_labels_do_not_change_earlier_folds(config):
    frame = development_frame(config)
    original = backtesting.rolling_folds(frame, config)[0]
    frame.loc[frame.FYEAR > 2010, "YIELD"] = 1e9
    changed = backtesting.rolling_folds(frame, config)[0]
    pd.testing.assert_frame_equal(original[0], changed[0])
    pd.testing.assert_frame_equal(original[1], changed[1])


@pytest.mark.parametrize("problem", ["duplicate", "missing_year", "missing_target"])
def test_invalid_development_data_is_rejected(config, problem):
    frame = development_frame(config)
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "missing_year":
        frame = frame.loc[frame.FYEAR != 2012]
    else:
        frame.loc[0, "YIELD"] = np.nan
    with pytest.raises(ValueError):
        backtesting.rolling_folds(frame, config)


def test_invalid_fold_boundaries_are_rejected(config):
    invalid = copy.deepcopy(config) | {"validation_end": 2009}
    with pytest.raises(ValueError):
        backtesting.rolling_folds(development_frame(config), invalid)
