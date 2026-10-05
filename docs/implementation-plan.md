# Crop Yield Prediction Implementation Plan

**Goal:** Build and verify a reproducible agricultural ML repository before publication.

**Architecture:** Three source CSVs become a validated county-year table of monthly weather and soil. Chronological model selection produces a versioned local artifact, evidence reports and a FastAPI demonstration.

**Tech Stack:** Python 3.12, pandas, scikit-learn, XGBoost, SHAP, matplotlib, FastAPI, pytest, Ruff.

## Global constraints

- Actual public data; official archive checksum; no fabricated metrics.
- IA, IL, IN, OH, MN, WI, MI, MO; 2000â€“2018; weather dekads 10â€“21.
- Train through 2013; validation 2014â€“2015; test 2016â€“2018; selection only on validation.
- Code MIT; source data CC BY 4.0; no secrets, raw data or pickle models in Git.
- Execute in this session, sequentially, to minimize resource use.

## Deliverables and checks

- [x] Acquisition: `src/crop_yield/data.py`, download MD5-checked archive and read required CSVs. `python -m crop_yield.cli download`. Check duplicate county-year/soil/dekad keys raise ValueError, corrupt archive raises ValueError.
- [x] Features: `src/crop_yield/features.py`, `build_table(raw_dir, config) -> (DataFrame, dict)`. Require three dekads per month, duration-weight TAVG, max/min extrema, sum PREC/ET0, compute balance. Join by county+year using one-to-one checks. Tests: future weather changes do not alter features; incomplete season is rejected; month sum and leap/calendar weighting match hand calculations.
- [x] Models: `src/crop_yield/modeling.py`, `temporal_split(table, config)` and `run_training(table, config, output_dir)`. Fixed baselines/RF/XGB, train-only preprocessing, validation selection, train+validation refit. Check year ranges are disjoint; perturb test targets cannot change selected model; county trend extrapolates a known simple line.
- [x] Evidence: `src/crop_yield/reporting.py`. Save validation/test metrics, held-out predictions, data audit, configuration, hashes, environment versions, plots and SHAP attribution. `python -m crop_yield.cli train`. Inspect residuals and test-year deterioration; no performance guarantee.
- [x] Inference: `src/crop_yield/predict.py`, `app/api.py`, `app/index.html`, `predict_rows(artifact, rows) -> list[dict]`. Validate exact schema, year/state, finite values and temperature ordering. `pytest -q` covers 422 inputs, 503 missing model and valid real artifact inference.
- [x] Documentation: README generated from executed results; data README, model card, French learning/interview guides, one compact executed notebook; Dockerfile and GitHub Actions. Check every local README link exists and notebook has no error output.
- [x] Release: Ruff, pytest, pip check, executed notebook, real HTTP prediction, fresh git clone in clean venv with exact requirements and full pipeline. Commit only reviewed deliverables. Create repo/description/topics and real improvement issues if authenticated publication is available; otherwise deliver verified archive and explain the precise access blocker.

Commands are exposed through the CLI; detailed interfaces and invariants live alongside tests rather than duplicating all source in this plan.

## Final status

Local deliverables, public repository, topics, three real issues and remote CI completed. Docker remains unavailable and unverified. See verification.md.
