import json

import pandas as pd
import pytest

from crop_yield.identity import candidate_crosswalk, county_reference
from crop_yield.identity_report import render_html


@pytest.fixture
def reference():
    return pd.DataFrame(
        [
            ("IA", "19", "001", "Adair County"),
            ("IA", "19", "141", "O'Brien County"),
            ("MO", "29", "189", "St. Louis County"),
            ("MO", "29", "510", "St. Louis city"),
            ("IL", "17", "001", "Adams County"),
            ("IA", "19", "003", "Adams County"),
        ],
        columns=["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"],
    )


def test_leading_zeros_punctuation_state_and_city_are_preserved(reference):
    actual = candidate_crosswalk(
        ["IL_ADAMS", "IA_ADAMS", "IA_O_BRIEN", "MO_ST_LOUIS", "MO_ST_LOUIS_CITY"], reference
    ).set_index("COUNTY_ID")
    assert actual.loc["IL_ADAMS", "fips"] == "17001"
    assert actual.loc["IA_ADAMS", "fips"] == "19003"
    assert actual.loc["IA_O_BRIEN", "county_ansi"] == "141"
    assert actual.loc["MO_ST_LOUIS", "fips"] == "29189"
    assert actual.loc["MO_ST_LOUIS_CITY", "fips"] == "29510"
    assert actual.status.eq("reference_match").all()


def test_unmatched_county_remains_unassigned(reference):
    row = candidate_crosswalk(["IA_MISSING"], reference).iloc[0]
    assert row.status == "unmatched" and row.candidate_count == 0
    assert row.fips == row.state_ansi == row.county_ansi == ""


def test_reference_normalization_collision_does_not_choose_a_code(reference):
    extra = pd.DataFrame([("IA", "19", "199", "O Brien County")], columns=reference.columns)
    row = candidate_crosswalk(["IA_O_BRIEN"], pd.concat([reference, extra])).iloc[0]
    assert row.status == "ambiguous_reference" and row.candidate_count == 2 and row.fips == ""


def test_historical_normalization_collision_cannot_create_a_many_to_one_identity(reference):
    with pytest.raises(ValueError, match="historical"):
        candidate_crosswalk(["IA_O_BRIEN", "IA_OBRIEN"], reference)


@pytest.mark.parametrize("identifier", [None, "", "IA_", "_ADAIR", "ia_ADAIR"])
def test_malformed_identifiers_fail_explicitly(identifier, reference):
    with pytest.raises(ValueError, match="identifier"):
        candidate_crosswalk([identifier], reference)


@pytest.mark.parametrize("code", [1, "1", "0010", None])
def test_county_codes_cannot_be_coerced_or_silently_zero_padded(code, reference):
    changed = reference.copy()
    changed["COUNTYFP"] = changed.COUNTYFP.astype(object)
    changed.loc[0, "COUNTYFP"] = code
    with pytest.raises(ValueError, match="code"):
        candidate_crosswalk(["IA_ADAIR"], changed)


def test_duplicate_reference_codes_and_empty_names_are_rejected(reference):
    with pytest.raises(ValueError, match="duplicate"):
        candidate_crosswalk(["IA_ADAIR"], pd.concat([reference, reference.iloc[:1]]))
    reference.loc[0, "COUNTYNAME"] = " "
    with pytest.raises(ValueError, match="name"):
        candidate_crosswalk(["IA_ADAIR"], reference)


def test_changed_reference_is_rejected_instead_of_repinning(tmp_path):
    manifest = tmp_path / "data/county-reference-manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"file": "county.txt", "sha256": "wrong", "bytes": 7}))
    raw = tmp_path / "data/raw"
    raw.mkdir()
    (raw / "county.txt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        county_reference(tmp_path)


def test_html_escapes_values_and_does_not_present_missing_counts_as_zero(reference):
    frame = candidate_crosswalk(["IA_ADAIR"], reference)
    frame.loc[0, "census_county_name"] = "<script>unsafe</script>"
    evidence = {
        "training_counties": 1,
        "matched_counties": 1,
        "training_rows": 2,
        "by_state": [{"state": "IA", "training_counties": 1, "matched_counties": 1}],
        "reference_sha256": "abc",
        "reference_url": "https://www2.census.gov/reference.txt",
        "scope": "Test fixture",
        "target_access_ready": False,
    }
    html = render_html(evidence, frame)
    assert "<script>unsafe</script>" not in html
    assert "&lt;script&gt;unsafe&lt;/script&gt;" in html
    assert "Couverture USDA non vérifiée" in html
    assert 'id="county-search"' in html
