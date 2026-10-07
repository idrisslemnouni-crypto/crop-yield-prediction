"""Target-free candidate county identities; no NASS target-access gate is passed."""

import json
import re
import urllib.request
from pathlib import Path

import pandas as pd

from crop_yield.data import file_hash
from crop_yield.features import build_table


def normalized_name(name):
    return re.sub(r"[^A-Z0-9]", "", re.sub(r" COUNTY$", "", name.upper().strip()))


def candidate_crosswalk(counties, reference):
    """Unique state/name matches only; preserve codes and refuse normalization collisions."""
    columns = ["STATE", "STATEFP", "COUNTYFP", "COUNTYNAME"]
    if not set(columns).issubset(reference.columns) or reference.empty:
        raise ValueError("Nonempty county reference schema required")
    reference = reference[columns].copy()
    for name, pattern in [
        ("STATE", r"[A-Z]{2}"),
        ("STATEFP", r"[0-9]{2}"),
        ("COUNTYFP", r"[0-9]{3}"),
    ]:
        if (
            not reference[name]
            .map(lambda v, p=pattern: isinstance(v, str) and bool(re.fullmatch(p, v)))
            .all()
        ):
            raise ValueError("Reference state and county codes must retain their string widths")
    if not reference.COUNTYNAME.map(
        lambda v: isinstance(v, str) and bool(normalized_name(v))
    ).all():
        raise ValueError("Reference county names must be nonempty")
    if reference.duplicated(["STATEFP", "COUNTYFP"]).any():
        raise ValueError("Reference contains duplicate geographic codes")
    if (
        reference.groupby("STATE").STATEFP.nunique().gt(1).any()
        or reference.groupby("STATEFP").STATE.nunique().gt(1).any()
    ):
        raise ValueError("Reference state-code mapping is not one-to-one")
    reference["normalized_name"] = reference.COUNTYNAME.map(normalized_name)
    rows, seen = [], set()
    if not counties:
        raise ValueError("Nonempty historical county identifiers required")
    for county in sorted(counties, key=str):
        if not isinstance(county, str) or not re.fullmatch(r"[A-Z]{2}_.+", county):
            raise ValueError("Malformed historical county identifier")
        state, name = county.split("_", 1)
        key = normalized_name(name)
        if not key:
            raise ValueError("Malformed historical county identifier")
        if (state, key) in seen:
            raise ValueError("Duplicate or colliding normalized historical identities")
        seen.add((state, key))
        candidates = reference.loc[reference.STATE.eq(state) & reference.normalized_name.eq(key)]
        row = dict(
            COUNTY_ID=county,
            state=state,
            normalized_name=key,
            candidate_count=len(candidates),
            status="unmatched",
            census_county_name="",
            state_ansi="",
            county_ansi="",
            fips="",
        )
        if len(candidates) == 1:
            match = candidates.iloc[0]
            row.update(
                status="reference_match",
                census_county_name=match.COUNTYNAME,
                state_ansi=match.STATEFP,
                county_ansi=match.COUNTYFP,
                fips=match.STATEFP + match.COUNTYFP,
            )
        elif len(candidates) > 1:
            row["status"] = "ambiguous_reference"
        rows.append(row)
    return pd.DataFrame(rows)


def county_reference(root):
    manifest = json.loads((root / "data/county-reference-manifest.json").read_text())
    if (
        Path(manifest["file"]).name != manifest["file"]
        or "/" in manifest["file"]
        or "\\" in manifest["file"]
    ):
        raise ValueError("Unsupported county reference filename")
    path = root / "data/raw" / manifest["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(manifest["url"], timeout=30) as response:
            content = response.read(1_000_001)
        if len(content) != manifest["bytes"] or len(content) > 1_000_000:
            raise ValueError("County reference size changed")
        temporary = path.with_suffix(".part")
        temporary.write_bytes(content)
        if file_hash(temporary) != manifest["sha256"]:
            temporary.unlink(missing_ok=True)
            raise ValueError("County reference checksum changed; review before repinning")
        temporary.replace(path)
    if path.stat().st_size != manifest["bytes"] or file_hash(path) != manifest["sha256"]:
        raise ValueError("County reference checksum changed; review before repinning")
    return pd.read_csv(path, sep="|", dtype=str, keep_default_na=False)


def run_identity(root: Path):
    """Reproduce historical features from pinned source bytes, then audit training IDs only."""
    from crop_yield.identity_report import render_html

    metrics = json.loads((root / "reports/metrics.json").read_text())
    config = json.loads((root / "configs/default.json").read_text())
    protocol = json.loads((root / "configs/future-evaluation.json").read_text())
    if config != metrics["config"] or set(config["states"]) != set(protocol["states"]):
        raise ValueError("Historical configuration differs from the recorded experiment/protocol")
    source = root / "data/raw"
    for name, expected in metrics["source_sha256"].items():
        if file_hash(source / name) != expected:
            raise ValueError("Historical source checksum differs from the recorded experiment")
    table, audit = build_table(source, config)
    if audit != metrics["data_audit"]:
        raise ValueError("Historical audit differs from the recorded experiment")
    first, last = protocol["training_years"]
    training = table.loc[table.FYEAR.between(first, last), ["COUNTY_ID", "FYEAR"]]
    if training.empty:
        raise ValueError("No historical training identities")
    result = candidate_crosswalk(training.COUNTY_ID.unique().tolist(), county_reference(root))
    counts = training.groupby("COUNTY_ID").agg(
        training_rows=("FYEAR", "size"), first_year=("FYEAR", "min"), last_year=("FYEAR", "max")
    )
    result = result.merge(counts, left_on="COUNTY_ID", right_index=True, validate="one_to_one")
    summary = (
        result.assign(matched=result.status.eq("reference_match"))
        .groupby("state", as_index=False)
        .agg(training_counties=("COUNTY_ID", "size"), matched_counties=("matched", "sum"))
    )
    manifest = json.loads((root / "data/county-reference-manifest.json").read_text())
    evidence = dict(
        protocol="county_identity_candidates_v1",
        training_years=[first, last],
        training_rows=len(training),
        training_counties=len(result),
        matched_counties=int(result.status.eq("reference_match").sum()),
        unmatched_counties=int(result.status.eq("unmatched").sum()),
        ambiguous_counties=int(result.status.eq("ambiguous_reference").sum()),
        by_state=summary.to_dict("records"),
        reference_url=manifest["url"],
        reference_sha256=manifest["sha256"],
        historical_source_sha256=metrics["source_sha256"],
        frozen_evaluation_sha256=file_hash(root / "configs/future-evaluation.json"),
        target_access_ready=False,
        new_targets_retrieved=False,
        new_scores_available=False,
        nass_count_status="Not executed; NASS API key required. Row counts would not establish usable target coverage.",
        scope="Candidate name/code correspondence only. NASS identity, boundary continuity, 2024-2025 observation coverage and predictor equivalence remain unverified.",
    )
    reports = root / "reports"
    result.to_csv(reports / "county-identity-crosswalk.csv", index=False)
    summary.to_csv(reports / "county-identity-summary.csv", index=False)
    (reports / "county-identity-audit.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    (reports / "county-identity.html").write_text(render_html(evidence, result), encoding="utf-8")
    return evidence
