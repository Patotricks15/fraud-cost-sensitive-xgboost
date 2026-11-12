# fraud-cost-sensitive-xgboost (FD02)

**Question:** How much does a cost-sensitive tree model improve on the FD01 logistic baseline?

## Portfolio

| ID | Repository | Focus |
|---|---|---|
| FD01 | [fraud-logistic-baseline](https://github.com/Patotricks15/fraud-logistic-baseline) | Logistic baseline |
| **FD02** | **fraud-cost-sensitive-xgboost** (this repo) | Cost-sensitive XGBoost |
| FD03 | [fraud-unsupervised-anomaly](https://github.com/Patotricks15/fraud-unsupervised-anomaly) | Isolation Forest and LOF |
| FD04 | [fraud-velocity-graph-features](https://github.com/Patotricks15/fraud-velocity-graph-features) | Velocity and graph features (blocked) |
| FD05 | [fraud-threshold-cost-policy](https://github.com/Patotricks15/fraud-threshold-cost-policy) | Cost-based threshold policy |
| FD06 | [fraud-drift-monitoring](https://github.com/Patotricks15/fraud-drift-monitoring) | Drift monitoring |

## Data

- Credit Card Fraud Detection, Kaggle `mlg-ulb/creditcardfraud` (real anonymized transactions, September 2013, 284,807 rows, 492 frauds). Same dataset as FD01.
- License per Kaggle API: Database Open License. Read date: 2026-11-08. The raw file is read in place and is not committed. See [data/README.md](data/README.md).
- `Time` covers about two days, so the time split spans two days only.
- The dataset has no categorical columns, so target encodings do not apply. Every feature is numeric.

## Method

- Features: `V1`-`V28` (PCA components, used as provided) and `log_amount = log1p(Amount)`. `Time` is excluded from features so that the model does not learn the time position.
- Split (time-ordered, no shuffle): earliest 70% of `Time` for training, next 15% for validation, last 15% for test.
- Baseline: `XGBClassifier` with `scale_pos_weight = n_neg / n_pos` computed on the training window only, fixed default-ish parameters (`max_depth=6`, `learning_rate=0.1`, `n_estimators=300`), `random_state=42`, `tree_method="hist"`, `eval_metric="aucpr"`.
- Tuned: random search over 30 configurations (`sklearn.model_selection.ParameterSampler`, `random_state=42`) on `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`, `min_child_weight` and `reg_lambda`. Each config is fit on training rows and selected by validation PR-AUC. The chosen config is refit on training rows only, so the test window is never used for selection or fitting.
- The search budget is 30 configs. This is a small random search, not an exhaustive one, so the tuned model may be below its best possible performance.
- Metrics: PR-AUC (primary), ROC-AUC, recall at a 1% alert rate, confusion matrix at threshold 0.5, Brier score. The test window is scored once.
- Feature importance is total gain from the tuned booster.

## Results

Real run (Kaggle data read from a local copy). Test window: 42,721 rows, 52 frauds. Validation: 42,718 rows, 56 frauds. Split: time-ordered, training 199,368 rows (384 frauds).

| Window | Model | PR-AUC | ROC-AUC | Recall @ 1% alerts | Brier |
|---|---|---|---|---|---|
| Validation | Baseline (fixed params) | 0.844 | 0.983 | 0.929 | not reported |
| Validation | Tuned (best of 30 configs) | 0.871 | 0.988 | 0.911 | not reported |
| Test | Baseline (fixed params) | 0.765 | 0.982 | 0.827 | 0.000442 |
| Test | Tuned (best of 30 configs) | 0.772 | 0.979 | 0.846 | 0.000453 |

Chosen config (validation PR-AUC 0.871): `max_depth=8`, `learning_rate=0.0348`, `n_estimators=463`, `subsample=0.837`, `colsample_bytree=0.717`, `min_child_weight=3`, `reg_lambda=1.068`.

Test confusion matrix at threshold 0.5: baseline TN 42,661, FP 8, FN 13, TP 39. Tuned TN 42,662, FP 7, FN 14, TP 38.

FD01 logistic test PR-AUC: 0.690 (from the FD01 README). Both XGBoost models beat it on test PR-AUC.

Top features by gain (tuned model): V14 (36%), V10 (16%), V12 (8%), V17 (6%), V4 (5%).

## Decision

Both tree models beat the FD01 logistic test PR-AUC (0.690) by a clear margin: baseline 0.765, tuned 0.772. The tuning gain over the fixed-parameter baseline is small on test: PR-AUC +0.007 and recall at 1% alerts +0.019, but ROC-AUC is slightly lower (0.979 vs 0.982). Its validation gain (+0.027 PR-AUC) did not carry fully to test. The tuned model is reported as the FD02 result because it was selected on validation only, but the test gain alone is too small to justify the search budget. The main gain over FD01 comes from the tree model itself.

The notebook's `FD01_TEST_PR_AUC` variable is still unset (its output says "not recorded yet"), so the comparison above uses the value from the FD01 README.

## Limitation

- Features are anonymized PCA components, so the model cannot be checked against business rules. The two-day window is too short to show seasonal or monthly drift.
- The search budget is 30 configurations, so the tuned result is a lower bound on what tuning could reach.
- `scale_pos_weight` shifts predicted probabilities upward, so the reliability diagram will show miscalibration. Scores from this model rank transactions well but are not calibrated probabilities.
- The downstream consumer (FD05) receives scores, not decisions, so no threshold is recommended here.

## Downstream use (FD05)

The notebook writes `data/fd02_scores.csv` with columns `split` (`val` or `test`), `Time`, `Amount`, `Class`, `score` (predicted fraud probability of the tuned model). It contains validation and test rows only. The tuned booster is saved to `data/fd02_xgb_tuned.json`.

Both files are git-ignored and not committed. The FD02 notebook was run locally against the downloaded copy. Copy `fd02_scores.csv` into the FD05 repository's `data/` folder (the FD05 notebook reads it from there).

## Reproduce

```bash
pip install -e ".[dev]"
pytest
# On Kaggle: attach mlg-ulb/creditcardfraud, upload the notebook, Run All.
# Locally (development only): export FRAUD_DATA_PATH=/path/to/creditcard.csv
make notebook
```

Figures are written to `reports/figures/`: `fd02_pr_curve.png`, `fd02_feature_importance.png`, `fd02_calibration.png`. Tables are written to `reports/`: `fd02_feature_importance.csv`, `fd02_search_trials.csv`.

## Layout

```text
src/fraud_xgb/   data loading, time split, metrics, model and search, score export, plots
notebooks/       jupytext source (.py) and paired notebook (.ipynb)
tests/           split, metrics, scale_pos_weight leakage, search and export tests
reports/         exported figures and tables
data/            README only; the raw data and FD02 exports are git-ignored
```
