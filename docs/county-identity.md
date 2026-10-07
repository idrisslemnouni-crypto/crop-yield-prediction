# Candidate county identities — 7 October 2026

The frozen 2024–2025 evaluation requires identity and predictor checks before inspecting reserved yields. This delivery resolves one narrower question: which official Census geographic codes correspond uniquely by state and name to the **historical training subset**? It does not complete the NASS identity gate.

## Observed result and method

The pinned historical source bytes reproduce the recorded data audit. The 2000–2015 training period supplies 8,133 observations across 528 distinct county identifiers. All 528 have one candidate Census reference match, with zero unmatched or ambiguous rows in this snapshot.

| State | Historical counties | Candidate matches |
|---|---:|---:|
| IA | 96 | 96 |
| IL | 98 | 98 |
| IN | 88 | 88 |
| MI | 38 | 38 |
| MN | 17 | 17 |
| MO | 76 | 76 |
| OH | 74 | 74 |
| WI | 41 | 41 |

Matching uses the state abbreviation and an uppercase county name after removing the trailing ` County` suffix and non-ASCII-alphanumeric separators. The city suffix is retained. For example, `IA_O_BRIEN` corresponds to `O'Brien County`; `IA_ADAIR` yields state code `19`, county code `001` and combined FIPS `19001`. Codes are strings, never numeric values with lost leading zeros.

There is no fuzzy matching or manual alias assignment. Duplicate/colliding historical names fail validation. Multiple matching reference names yield `ambiguous_reference`; no match yields `unmatched`, with empty geographic-code fields. The published state totals describe this historical subset, not all counties or agricultural coverage in those states.

## Sources and usage

- Historical inputs: Paudel, de Wit and Boogaard, [Zenodo 7751191](https://doi.org/10.5281/zenodo.7751191), CC BY 4.0. Original checksums remain in `reports/metrics.json`.
- Geographic reference: [U.S. Census Bureau national county codes](https://www2.census.gov/geo/docs/reference/codes2020/national_county2020.txt), downloaded 7 October 2026; 126,639 bytes, 3,235 reference rows; SHA-256 `9f6e5f6eb6ac2f5e9a36d5fd01dec77991bddc75118f748a069441a4782970d6`. [Official ANSI-code explanation](https://www.census.gov/library/reference/code-lists/ansi.html) and [Census citation/public-use guidance](https://www.census.gov/about/policies/citation.html). This is a separately attributed U.S. government reference; no blanket worldwide CC0 designation is asserted.

The exact reference bytes are pinned in `data/county-reference-manifest.json`. A changed download or local cache fails the checksum check and requires source review before repinning. Raw files remain ignored. The `codes2020` directory and HTTP Last-Modified date (13 February 2023) do not establish unchanged boundaries from 2000 through 2025. The crosswalk contains historical-source identifiers plus Census candidate codes; retain attribution to both sources when reusing it.

## Reproduction and checks

After installing the pinned Python environment, run `python -m crop_yield.cli download` and `python -m crop_yield.cli identity-audit`. The latter verifies historical hashes/configuration and reproduces the data audit before writing separate identity CSV/JSON/HTML reports. It performs no model fitting and does not replace prior reports, model bytes or the frozen evaluation configuration.

`notebooks/02_county_identity.ipynb` contains three executed code cells: actual acquisition/audit, descriptive state counts and historical code checks. The HTML report opens offline and supports case-insensitive filtering. `Rscript scripts/check_county_identity.R` independently checks the committed CSVs with base R: code widths/uniqueness, state/county concatenation, and recomputed state totals. It does not call Python, train or download yields. R is unavailable locally; the separate GitHub job provides remote execution evidence when successful.

## Remaining gates

USDA NASS metadata/count access has **not been executed**: no API key is available in this environment. The official [Quick Stats API documentation](https://quickstats.nass.usda.gov/api) describes count/parameter endpoints separately from data retrieval. Future access should use a locally configured key without committing or printing it. A count alone would not establish usable numeric county-year coverage because suppressed/aggregate/duplicate records and product definitions still need checks.

NASS code/name equivalence and geographic boundary continuity remain unverified. Historical predictor equivalence to future AgERA5 acquisition remains unverified. Reserved 2024–2025 yields have not been retrieved, and no new evaluation scores exist. The prior test has already been inspected; this audit neither resets its status nor improves the selected model's documented failure to generalize. Development used substantial AI assistance; no field validation or independent personal mastery is claimed.
