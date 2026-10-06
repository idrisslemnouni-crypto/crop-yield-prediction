# Independent-data audit and frozen evaluation plan

Metadata audit on 6 October 2026. **No new evaluation or score has been produced.** The machine-readable [protocol](../configs/future-evaluation.json) reserves 2024–2025 county maize yields; target retrieval is blocked until its feature/identity gates pass.

## Source decision

[USDA NASS Quick Stats](https://quickstats.nass.usda.gov/api) provides the candidate county-year targets and requires an API key. Its [federal catalogue entry](https://catalog.data.gov/dataset/quick-stats-agricultural-database) declares public access and CC0. Select annual survey county grain-corn yield, total domain, all production practices, bushels/acre. Suppressed values and non-county aggregates must be excluded with coverage counts. The [2024 programme reinstatement notice](https://www.nass.usda.gov/Newsroom/Notices/2025/03-19-2025.php) establishes a release route; exact coverage for both reserved years and the eight states still needs a metadata/count check. State/county ANSI codes must retain leading zeros.

The authors' [US data card](https://github.com/WUR-AI/AgML-CY-Bench/blob/main/data_preparation/crop_statistics_US/README.md) describes a useful preparation reference, but its NASS targets are the same underlying source. [CY-Bench v1.10](https://zenodo.org/records/17279151) does not independently validate the original benchmark or establish a ready 2024–2025 test. Component licences and meteorology-version changes require their own checks.

## Predictor compatibility and blockers

[Copernicus AgERA5 metadata](https://cds.climate.copernicus.eu/datasets/sis-agrometeorological-indicators?tab=overview) documents daily temperature, precipitation and reference evapotranspiration. Availability alone does not make a new extraction equivalent to the original dekadal county summaries. Freeze the product/version, spatial aggregation, temperature extrema and accumulation rules after an overlap check on historical data. Verify the county crosswalk before reusing the original static soil scale. Use April–July only; fit neither imputation nor scaling on evaluation features.

The plan fixes historical training at 2000–2015 and compares the original selected XGBoost against a training-only county trend. The residual learner stays unpromoted. Freeze eligible counties, exclusions and prediction files before joining reserved labels. Report annual/pooled errors and retained coverage once; do not select a model from that evaluation. Reanalysis-based evaluation remains retrospective until historical July availability is established.

## Exposure and existing evidence

All original source/test results already inspected remain historical evidence. During metadata research, snippets unexpectedly showed some **2023 Iowa, Minnesota and Wisconsin yields**. That exposure is recorded; 2023 will not be called untouched. No target files were downloaded and no 2024–2025 target values appeared during this audit. This statement describes the audit, rather than guaranteeing what future users have seen.

Original model and final-test reports remain unchanged. Development/residual diagnostics are documented separately. The next useful action is a metadata/count and predictor-equivalence check, followed by a protocol revision if a gate fails, **before** accessing new targets. Assistance IA substantial; no personal mastery or operational/field performance is inferred.
