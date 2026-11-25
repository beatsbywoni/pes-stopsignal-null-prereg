# Preregistration (Study 3): slowing after failed stops relative to a race-calibrated null

This repository registers, before analysis, the hypotheses, measures, sample, decision rules, and analysis code of a test in the holdout of the Self-Regulation Ontology stop-signal data (participants of the first wave who did not take part in the retest). It follows a reanalysis of Hedge et al. (2018) and an earlier registration (https://doi.org/10.5281/zenodo.24534671).

- `PREREGISTRATION_study3_holdout.md`: the registration (version 1.0).
- `code/study3_holdout.py`: the analysis; the other modules in `code/` are identical to those deposited with the earlier registration.
- `code/holdout_counts.py` and `holdout_counts.json`: counts of the holdout that use no response time (what was known).
- `synthetic_test.json`: output of the code on synthetic data (every response time replaced by a random draw), produced before registration.
- `requirements.txt`, `SHA256SUMS.txt`.

Data: https://github.com/IanEisenberg/Self_Regulation_Ontology, folder `Data/`, commit 4e6e837fb5aaca87ed5914fd9be251156f43eec9.

Run: `python3 code/study3_holdout.py <path to Data> results.json` (and `--synthetic` for the test).
