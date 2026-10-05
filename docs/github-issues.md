# Open research issues

Three genuine unfinished tasks were created as [issues 1–3](https://github.com/idrisslemnouni-crypto/crop-yield-prediction/issues). None has been closed or represented as completed. Remote CI now passes; the third live issue concerns the container alone.

## 1. Add forward-chaining validation and an untouched test period

The validation winner underperforms the county-trend baseline on 2016–2018. These test years have now been inspected and must not be reused as an independent test for a tuned replacement.

Acceptance criteria: document several chronological development folds; fit all preprocessing/trends inside each fold; register candidate rules before evaluating a newly acquired untouched period or territory; save every fold score, data provenance and final predictions. Investigate county-trend plus residual weather modeling only within development folds. Preserve the original benchmark evidence.

## 2. Confirm soil feature units and quantify coverage bias

The source archive supplies SM_WHC, but the Zenodo landing description does not specify its exact scale/unit. The soil join excludes 2,541 selected labels.

Acceptance criteria: locate and cite authoritative source processing metadata for SM_WHC; compare retained/excluded county coverage and yield histories using available public identifiers; document whether exclusions change representativeness; do not convert SM_WHC into mm or irrigation demand without a verified convention.

## 3. Verify container and remote CI

The Windows environment has no Docker executable and GitHub publication lacks authentication. Local tests and fresh-clone training pass; remote CI/container execution remain unverified.

Acceptance criteria: run GitHub Actions on the published repository; build the supplied Dockerfile; mount the trusted locally trained model read-only; verify /health, real prediction and invalid-input behavior; record successful commands and actual logs. Add no production deployment claim without an actual deployment.
