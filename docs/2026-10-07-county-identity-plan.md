# County identity audit implementation plan

**Goal:** build a target-free, reproducible candidate crosswalk from historical training county names to official Census codes, while preserving the frozen future-evaluation gates.

**Architecture:** pin the small public Census reference in an ignored source cache; normalize county names conservatively within state and require unique matches. Separate pure mapping contracts from acquisition/report generation. Produce CSV/JSON, an executed Jupyter notebook and a browser-readable HTML report without training a model or retrieving reserved yield values.

- [x] Download the official county-code reference, record URL/date/size/hash and usage source, and probe match coverage on training-year identifiers only.
- [x] Test leading zeros, cross-state matching, city/county distinction, normalization collisions, missing/malformed keys, explicit unmatched counties and reference revision detection.
- [x] Implement the candidate crosswalk with documented normalization and explicit non-ready target-access status; provide an independent base-R CI check. Keep NASS count access and predictor equivalence open.
- [x] Execute the real audit and notebook; inspect the HTML. Preserve all existing scientific outputs/model bytes and attribute both input sources.
- [x] Complete local checks (67 passed), inspect the implementation and stage the bounded delivery.
- [ ] Commit/push, scan the new commit and verify the exact-SHA GitHub workflow, including R.
- [ ] Update only this repository's evidence/archive and the private daily backlog; retain the completed ten-repository session and the 10:00 Morocco schedule. Post-publication completion evidence belongs in the external daily journal, without a commit solely to check these boxes.

No fuzzy/manual alias mapping, new scoring, target-value retrieval or expensive training is in scope. A Census name/code match does not establish NASS identity, stable historical boundaries, or future eligible observation coverage. The small R check validates geographic string codes and descriptive counts independently; no local R execution is claimed while that runtime is absent.
