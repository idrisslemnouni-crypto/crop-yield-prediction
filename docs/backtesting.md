# Development backtesting protocol

The previously inspected final test (2016–2018) cannot become an untouched holdout again. This extension diagnoses stability on development data, without revising original final-test results, choosing a new winner or saving a replacement model.

Six expanding folds: train 2000–2009 → validate 2010; then expand training by one year until train 2000–2014 → validate 2015. Counties can recur. This is temporal transfer in covered US counties, not independent spatial validation or operational forecast-vintage evaluation.

Candidates are fresh scikit-learn clones for each fold: global mean, county trend, Ridge alpha 10, Random Forest and XGBoost. Tree settings match the original protocol. No hyperparameter search. Tree imputation and state encoding are training-only; Ridge additionally scales the encoded training features. County trends extrapolate from historical labels and use a global fallback for unseen counties.

Validation yields enter metrics only, not preprocessing or fitting. FYEAR 2016 onward is excluded before fold construction and diagnostic data hashing. Duplicate county-year keys, incomplete annual coverage and non-finite development targets are rejected. Tests poison the final-test labels/features and show all diagnostics remain identical; they also check chronological disjointness and that later development labels cannot alter earlier folds.

Source: Paudel, de Wit and Boogaard, Zenodo 7751191, CC BY 4.0, with the original archive checksum and attribution in data/README.md. The development hash includes only retained development features/labels in sorted county-year order. Dependency versions are pinned in requirements-lock.txt.

Run from the repository with Python 3.12:

```bash
python -m crop_yield.cli backtest
python -m pytest -q
```

Outputs: reports/development_backtest.json (boundaries, training/evaluation counts, fixed model representations, seed, per-fold and pooled errors), reports/development_predictions.csv, reports/figures/development-backtest.png. RMSE/MAE unit: bushels/acre. Pooled R² uses all development validation observations; it is not the mean of per-year scores. Years and counties are dependent, and six folds do not establish robust uncertainty.

## Executed results

| Model | Pooled development RMSE | MAE | R² |
|---|---:|---:|---:|
| county_trend | 34.74 | 26.20 | 0.007 |
| global_mean | 35.63 | 28.20 | -0.044 |
| random_forest | 30.75 | 23.49 | 0.222 |
| ridge | 33.44 | 25.63 | 0.080 |
| xgboost | 28.43 | 21.82 | 0.335 |

XGBoost has lower pooled development error, while county trend still beat it on the original final test. This difference motivates drift investigation, not a claim of improved forecasting. The [training-only residual experiment](residual-learning.md) has now been executed with separate outputs: its pooled RMSE is 31.997, worse than raw-yield XGBoost's 28.432. These original baseline reports are preserved. Next task: identify a genuinely new future/geographic dataset and freeze its protocol before validating a revised model. All code and interpretation are AI-assisted; field validity is unverified.
