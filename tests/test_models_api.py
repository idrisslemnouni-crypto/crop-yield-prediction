import copy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import create_app
from crop_yield.features import numeric_features
from crop_yield.modeling import CountyTrend, temporal_split, tree_pipeline
from crop_yield.predict import predict_rows, validate_rows


def example_row():
    row = {"COUNTY_ID": "IA_TEST", "STATE": "IA", "FYEAR": 2016, "SM_WHC": 14.0}
    for month in (4, 5, 6, 7):
        row.update(
            {
                f"{name}_{month:02d}": value
                for name, value in [
                    ("tavg", 20.0),
                    ("tmax", 30.0),
                    ("tmin", 10.0),
                    ("prec", 30.0),
                    ("et0", 45.0),
                    ("balance", -15.0),
                ]
            }
        )
    return row


@pytest.fixture
def artifact(config):
    # Artificial fixtures verify contracts only; never used for reported research metrics.
    train = pd.DataFrame({"COUNTY_ID": ["IA_TEST"] * 4, "FYEAR": [2010, 2011, 2012, 2013]})
    model = CountyTrend().fit(train, pd.Series([100.0, 102.0, 104.0, 106.0]))
    return {
        "model": model,
        "selected_model": "county_trend",
        "config": config,
        "numeric_features": numeric_features(config),
        "known_counties": ["IA_TEST"],
        "training_years": [2010, 2013],
    }


def test_county_trend_extrapolation_and_unseen_fallback(artifact):
    frame = pd.DataFrame({"COUNTY_ID": ["IA_TEST", "IA_UNKNOWN"], "FYEAR": [2016, 2016]})
    np.testing.assert_allclose(artifact["model"].predict(frame), [112, 112], atol=1e-10)


def test_temporal_splits_are_disjoint(config):
    frame = pd.DataFrame({"FYEAR": list(range(2000, 2019)), "YIELD": list(range(19))})
    train, val, test = temporal_split(frame, config)
    assert (train.FYEAR.max(), val.FYEAR.min(), val.FYEAR.max(), test.FYEAR.min()) == (
        2013,
        2014,
        2015,
        2016,
    )
    assert set(train.FYEAR).isdisjoint(test.FYEAR)
    assert len(train) + len(val) + len(test) == len(frame)


def test_empty_temporal_split_is_rejected(config):
    with pytest.raises(ValueError, match="non-empty"):
        temporal_split(pd.DataFrame({"FYEAR": [2000, 2016]}), config)


def test_model_preprocessing_never_sees_target_or_county(config):
    pipeline = tree_pipeline(config, "xgboost")
    inputs = [
        name
        for _, _, columns in pipeline.named_steps["preprocess"].transformers
        for name in columns
    ]
    assert "YIELD" not in inputs
    assert "COUNTY_ID" not in inputs
    assert "STATE" in inputs


def test_inference_roundtrip_is_stable(artifact, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    assert predict_rows(joblib.load(path), [example_row()])[0]["yield_bu_ac"] == pytest.approx(112)


@pytest.mark.parametrize(
    "field,value",
    [
        ("FYEAR", 2026),
        ("FYEAR", 2016.5),
        ("FYEAR", True),
        ("STATE", "MA"),
        ("COUNTY_ID", "IL_TEST"),
        ("SM_WHC", -1),
        ("SM_WHC", float("nan")),
        ("prec_04", -1),
        ("balance_04", 0),
        ("tmin_04", 50),
        ("tavg_04", "20"),
    ],
)
def test_invalid_inference_is_rejected(artifact, field, value):
    row = example_row()
    row[field] = value
    with pytest.raises(ValueError):
        validate_rows(artifact, [row])


def test_target_leakage_in_request_is_rejected(artifact):
    row = example_row()
    row["YIELD"] = 200
    with pytest.raises(ValueError, match="exact fields"):
        validate_rows(artifact, [row])


def test_api_missing_model_is_503(tmp_path):
    with TestClient(create_app(tmp_path / "absent.joblib")) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/predict", json={"rows": [example_row()]}).status_code == 503


def test_api_prediction_and_validation(artifact, tmp_path):
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    with TestClient(create_app(path)) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/predict", json={"rows": [example_row()]})
        assert response.status_code == 200
        assert response.json()["predictions"][0]["yield_bu_ac"] == pytest.approx(112)
        invalid = copy.deepcopy(example_row())
        invalid["balance_04"] = 100
        assert client.post("/predict", json={"rows": [invalid]}).status_code == 422
        invalid["FYEAR"] = 2026
        assert client.post("/predict", json={"rows": [invalid]}).status_code == 422
        assert client.post("/predict", json={"rows": []}).status_code == 422
        assert client.get("/").status_code == 200


def test_real_artifact_inference_when_available():
    root = Path(__file__).parents[1]
    model = root / "models" / "model.joblib"
    if not model.exists():
        pytest.skip("Real model built by the separate reproduction command")
    import json

    from crop_yield.predict import load_artifact

    artifact = load_artifact(model)
    row = json.loads((root / "reports" / "example-input.json").read_text())
    assert np.isfinite(predict_rows(artifact, [row])[0]["yield_bu_ac"])
