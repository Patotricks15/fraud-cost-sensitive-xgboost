# fraud-cost-sensitive-xgboost (FD02)

**Question:** How much does a cost-sensitive tree model improve on the FD01 logistic baseline?

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

Not yet produced. The notebook needs the Kaggle dataset, which is not downloaded to this workstation. After running the notebook on Kaggle, fill in these tables:

| Window | Model | PR-AUC | ROC-AUC | Recall @ 1% alerts | Brier |
|---|---|---|---|---|---|
| Validation | Baseline (fixed params) | pending | pending | pending | pending |
| Validation | Tuned (best of 30 configs) | pending | pending | pending | pending |
| Test | Baseline (fixed params) | pending | pending | pending | pending |
| Test | Tuned (best of 30 configs) | pending | pending | pending | pending |

Test window confusion matrix at threshold 0.5: pending.

FD01 logistic test PR-AUC for comparison: pending (record from the FD01 notebook output).

## Decision

Pending the Kaggle run. The tuned tree model must beat the FD01 logistic test PR-AUC to justify its extra complexity. If it does not, FD01 remains the reference model. The notebook variable `FD01_TEST_PR_AUC` holds that value for the comparison.

## Limitation

- Features are anonymized PCA components, so the model cannot be checked against business rules. The two-day window is too short to show seasonal or monthly drift.
- The search budget is 30 configurations, so the tuned result is a lower bound on what tuning could reach.
- `scale_pos_weight` shifts predicted probabilities upward, so the reliability diagram will show miscalibration. Scores from this model rank transactions well but are not calibrated probabilities.
- The downstream consumer (FD05) receives scores, not decisions, so no threshold is recommended here.

## Downstream use (FD05)

The notebook writes `data/fd02_scores.csv` with columns `split` (`val` or `test`), `Time`, `Amount`, `Class`, `score` (predicted fraud probability of the tuned model). It contains validation and test rows only. The tuned booster is saved to `data/fd02_xgb_tuned.json`.

Both files are git-ignored and not committed. After the Kaggle run, copy `fd02_scores.csv` from the Kaggle output into the FD05 repository's `data/` folder.

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
