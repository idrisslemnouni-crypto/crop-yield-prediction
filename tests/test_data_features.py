import copy

import pandas as pd
import pytest

from crop_yield.data import verify_archive
from crop_yield.features import aggregate_weather, build_table, unique_keys


def test_corrupt_archive_is_rejected(tmp_path):
    path = tmp_path / "bad.zip"
    path.write_bytes(b"not the source archive")
    with pytest.raises(ValueError, match="checksum"):
        verify_archive(path)


def test_monthly_total_and_weighted_mean(weather, config):
    weather.loc[weather.DEKAD == 12, "TAVG"] = 26.0
    weather.loc[weather.DEKAD == 15, "TAVG"] = 26.0
    result = aggregate_weather(weather, config).iloc[0]
    assert result.prec_04 == 30
    assert result.et0_04 == 45
    assert result.balance_04 == -15
    assert result.tavg_04 == pytest.approx(22)
    assert result.tavg_05 == pytest.approx((20 * 20 + 26 * 11) / 31)


def test_post_cutoff_weather_cannot_change_features(weather, config):
    changed = weather.copy()
    changed.loc[changed.DEKAD > 21, ["TAVG", "TMAX", "TMIN", "PREC", "ET0"]] = 9999.0
    pd.testing.assert_frame_equal(
        aggregate_weather(weather, config), aggregate_weather(changed, config)
    )


def test_incomplete_season_is_excluded(weather, config):
    assert aggregate_weather(weather.loc[weather.DEKAD != 11], config).empty


def test_duplicate_weather_key_is_rejected(weather, config):
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_weather(pd.concat([weather, weather.iloc[[0]]]), config)


@pytest.mark.parametrize(
    "column,value", [("PREC", -1), ("ET0", -1), ("TAVG", 100), ("TMAX", float("nan"))]
)
def test_invalid_weather_is_rejected(weather, config, column, value):
    weather.loc[weather.DEKAD == 10, column] = value
    with pytest.raises(ValueError):
        aggregate_weather(weather, config)


def test_invalid_dekad_is_rejected(weather, config):
    weather.loc[0, "DEKAD"] = 37
    with pytest.raises(ValueError, match="DEKAD"):
        aggregate_weather(weather, config)


def test_duplicate_soil_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        unique_keys(pd.DataFrame({"COUNTY_ID": ["IA_TEST", "IA_TEST"]}), ["COUNTY_ID"], "soil")


def test_join_audit_accounts_for_missing_soil(weather, config, tmp_path):
    pd.DataFrame(
        {"COUNTY_ID": ["IA_TEST", "IA_OTHER"], "FYEAR": [2000, 2000], "YIELD": [150.0, 120.0]}
    ).to_csv(tmp_path / "YIELD_COUNTY_US.csv", index=False)
    pd.DataFrame({"COUNTY_ID": ["IA_TEST"], "SM_WHC": [14.0]}).to_csv(
        tmp_path / "SOIL_COUNTY_US.csv", index=False
    )
    weather.to_csv(tmp_path / "METEO_COUNTY_US.csv", index=False)
    table, audit = build_table(tmp_path, config)
    assert len(table) == 1
    assert audit["dropped_without_soil"] == 1
    assert audit["source_weather_rows"] == 36
    assert audit["dropped_without_complete_weather"] == 0


def test_benchmark_cutoff_contract_is_explicit(config, tmp_path):
    changed = copy.deepcopy(config)
    changed["cutoff_dekad"] = 27
    with pytest.raises(ValueError, match="contract"):
        build_table(tmp_path, changed)
