# Model card — chronological maize benchmark

## Intended use

Research and educational demonstration of county-level maize-yield regression and ML engineering. The API serves the validation-selected XGBoost model for historical examples from eight US states, 2000–2018. It is not recommended for operational yield, insurance, irrigation or policy decisions.

## Training and selection

Paudel/de Wit/Boogaard public county dataset, Zenodo 7751191, CC BY 4.0. Training candidates: global mean, county trend, Random Forest, XGBoost. Initial train 2000–2013, validation 2014–2015; selection by validation RMSE. Selected XGBoost is refit on 2000–2015. Final test is 2016–2018, without model reselection after inspection. Exact hyperparameters, environment versions, source/model hashes and sizes are in `reports/metrics.json` and code.

Inputs: year, soil capacity, state, monthly April–July temperature summaries, precipitation, ET0 and balance. COUNTY_ID is retained for identity validation and the baseline trend but is excluded from tree features. Source data availability is retrospective reanalysis; historical forecast-vintage availability is not established.

## Actual evaluation

XGBoost validation RMSE 22.88 bu/ac; final test RMSE 33.21, MAE 28.27, R² −0.650. County-trend baseline test RMSE 26.79. All candidates have negative test R². There are 1,391 held-out county-years but only three year clusters. County observations share climate and are not independent. Per-state/year errors and every held-out prediction are saved. The observed model failure is the principal conclusion.

## Risks and boundaries

- Soil availability reduces coverage from 12,065 selected labels to 9,524 rows, so excluded counties may differ systematically.
- Trees extrapolate yield trends poorly. Strong year attribution alone does not prove the causal mechanism of failure.
- No field-level, Morocco, unseen-state or unseen-county validation. All final-test counties occurred in final training.
- Static soil scale follows the source; its exact SM_WHC unit is not documented on the landing page. No derived irrigation quantity is asserted.
- No calibrated predictive uncertainty. The three-year bootstrap interval is descriptive and fragile.
- AI-assisted code and documentation. Human mastery must be demonstrated separately.

## Operational behavior

Missing model: HTTP 503. Invalid/extra fields, unsupported state/year, unknown county or incoherent monthly balance: HTTP 422. Maximum 100 rows per request. No authentication, rate limiting, model registry or production monitoring is claimed. The model is a trusted local joblib artifact, never an arbitrary downloaded pickle. API tests verify contracts, not scientific suitability.

## Future evaluation

Do not retune against the already-inspected 2016–2018 test and call it independent. Define forward-chaining development folds, introduce training-only detrending if justified, and evaluate against new untouched years or territories. Then reassess operational vintages, geographic transfer and calibration.
