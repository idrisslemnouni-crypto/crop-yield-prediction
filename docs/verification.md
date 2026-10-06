# Verification — 5 October 2026

Subsequent delivery, 6 October 2026: six expanding development folds and a separate training-only residual experiment, 51 local tests / 50 CI tests plus one absent-artifact skip at `e9cddca`, original reports/model preserved. A further metadata-only future-data audit adds a frozen 2024–2025 protocol without new target retrieval, training or score claims. Earlier counts below describe the original release checkpoint.

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

## Publication and remaining limits

- Docker executable unavailable; container build/run not verified.
- Repository published publicly at [idrisslemnouni-crypto/crop-yield-prediction](https://github.com/idrisslemnouni-crypto/crop-yield-prediction), default branch main, with six relevant topics. [GitHub Actions run 37248712740](https://github.com/idrisslemnouni-crypto/crop-yield-prediction/actions/runs/37248712740) succeeded: lint, formatting, dependency check, notebook validation, and 32 tests passed / 1 real-artifact test skipped because the artifact is not committed. The full 33-test check and training were verified locally from a clean clone. Three genuine research/container issues are open; no unfinished issue was closed.
- CV, personal mastery, academic eligibility, field performance, Moroccan transfer and current operational forecasting are not verified.

Source archive and Git repository are available locally and publicly. No profile pin or production deployment is claimed. The publication script is for initial creation only; the existing origin must be used for subsequent pushes.

## 5 October 2026 development extension

The real-data `backtest` command completed on all six development folds. Original model and final-test reports were not regenerated. Local pytest: 40 passed, including seven new chronology/holdout contracts; warnings concern the existing Starlette test client and Windows temporary-folder cleanup. New GitHub CI evidence is recorded externally after the pushed commit completes.
