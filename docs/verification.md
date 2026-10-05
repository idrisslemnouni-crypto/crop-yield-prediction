# Verification — 5 October 2026

## Verified locally

- Official archive downloaded, exact byte count and official MD5 matched; three source CSVs extracted. Source SHA-256 hashes recorded in reports/metrics.json.
- Full training executed on 9,524 genuine county-years from 528 counties, with fixed chronological splits. Every reported metric derives from executed predictions.
- A fresh Git clone and a new isolated Python 3.12 virtual environment installed requirements-lock.txt and the editable package successfully.
- The fresh clone downloaded the source independently and ran the full training pipeline. Selection and source hashes matched; validation/test metrics and all saved predictions matched within 1e-10 absolute and 1e-12 relative floating-point tolerance. One Random Forest validation MAE differed by about 5e-15 between runs, consistent with parallel floating-point summation.
- pytest: **33 passed** in the project and in the fresh clone, including the real-artifact inference check. No skipped test once the artifact was trained.
- Ruff lint and formatting checks passed; modules compiled; pip check found no broken dependencies.
- Notebook: five code cells executed successfully; notebook format validated; no error output; all model RMSE values recomputed from held-out predictions.
- HTTP /health: 200, xgboost, training years 2000–2015. HTTP /predict on the real IA_ADAIR / 2016 example returned **152.56544494628906 bu/acre**. TestClient verifies invalid input returns 422 and an absent model returns 503.
- The browser demo loaded the real example and produced the same prediction. Screenshot saved in reports/figures/app-demo.jpg. All five scientific figures were visually inspected.
- Local Markdown links, common credential patterns, notebook portability and tracked-file sizes checked. No raw/processed dataset, joblib model, .env secret or file over 5 MB is tracked.

## Material findings

The code is reproducible, but the selected model fails to generalize adequately: test R² −0.650; test RMSE 33.21 bu/ac versus the county-trend baseline's 26.79. This is explicitly documented. A new model must not be tuned on the already-inspected test and represented as independently evaluated.

Non-blocking warning: installed Starlette deprecates the current httpx TestClient transport in favor of httpx2. Tests pass in both environments. Windows pytest temporary-directory cleanup warnings appeared in one local repeat; no project test failed. No warning is treated as a passed remote or operational check.

## Not verified / not completed

- Docker executable unavailable; container build/run not verified.
- GitHub CLI has no authenticated host; in-app GitHub browser shows login; the connector returns no accessible repositories and provides no repository-creation tool. **New repository not published.** No remote Actions run, topics, pinned repository or GitHub issue is claimed.
- CV, personal mastery, academic eligibility, field performance, Moroccan transfer and current operational forecasting are not verified.

Source archive and Git repository are ready locally. scripts/publish.ps1 checks the authenticated owner and clean repository before creating/pushing the intended public repository. Prepared genuine future issues are listed in docs/github-issues.md.
