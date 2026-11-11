# %% [markdown]
# # FD02: Cost-sensitive XGBoost
# Source for the FD02 notebook. Convert with `jupytext --to ipynb` and run on Kaggle, or set `FRAUD_DATA_PATH` locally.
# Dataset: Credit Card Fraud Detection (`mlg-ulb/creditcardfraud`), time-ordered split 70/15/15.
# Train on the earliest 70% of time, tune on the next 15%, report the last 15% once.
# Question: how much does a cost-sensitive tree model improve on the FD01 logistic baseline?

# %%
from pathlib import Path

import pandas as pd

from fraud_xgb import plots
from fraud_xgb.data import load_creditcard, time_split
from fraud_xgb.export import build_score_export, write_score_export
from fraud_xgb.metrics import brier, confusion_at_threshold, metric_summary
from fraud_xgb.model import (
    FEATURES,
    add_features,
    build_baseline,
    build_tuned,
    gain_importance,
    scale_pos_weight,
    search,
)

IN_NOTEBOOKS = Path.cwd().name == "notebooks"
FIG_DIR = Path("../reports/figures") if IN_NOTEBOOKS else Path("reports/figures")
REPORT_DIR = Path("../reports") if IN_NOTEBOOKS else Path("reports")
DATA_DIR = Path("../data") if IN_NOTEBOOKS else Path("data")
ALERT_RATE = 0.01
N_SEARCH_CONFIGS = 30
# Record the FD01 logistic test PR-AUC from the FD01 notebook output before the comparison is read.
FD01_TEST_PR_AUC = None
DATASET = "creditcardfraud"
BASELINE_LABEL = "XGBoost baseline (scale_pos_weight, fixed params)"
TUNED_LABEL = f"XGBoost tuned ({N_SEARCH_CONFIGS} random configs)"

# %%
df = add_features(load_creditcard())
split = time_split(df)
X, y = df[FEATURES], df["Class"]
print(f"rows: {len(df):,}  frauds: {int(y.sum())}")
print(f"train end (Time): {split.train_end:,.0f}s  val end: {split.val_end:,.0f}s")
for name, mask in (("train", split.train), ("val", split.val), ("test", split.test)):
    print(f"{name}: {int(mask.sum()):,} rows, {int(y[mask].sum())} frauds")

# %%
spw = scale_pos_weight(y[split.train])
print(f"scale_pos_weight (train only): {spw:.3f}")

# %% [markdown]
# ## Baseline on validation (fixed parameters, no tuning)

# %%
baseline = build_baseline(spw).fit(X[split.train], y[split.train])
baseline_val = baseline.predict_proba(X[split.val])[:, 1]
print(metric_summary(y[split.val], baseline_val, ALERT_RATE))

# %% [markdown]
# ## Random search on validation PR-AUC only
# Each config is fit on the training window and scored on validation. The test window is not touched.

# %%
best_params, trials = search(
    X[split.train], y[split.train], X[split.val], y[split.val], n_iter=N_SEARCH_CONFIGS
)
print(trials.sort_values("val_pr_auc", ascending=False).head(5).to_string(index=False))
print(f"chosen config: {best_params}")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
trials.to_csv(REPORT_DIR / "fd02_search_trials.csv", index=False)

# %% [markdown]
# ## Refit the chosen config on training rows only

# %%
tuned = build_tuned(best_params, spw).fit(X[split.train], y[split.train])
tuned_val = tuned.predict_proba(X[split.val])[:, 1]
print(metric_summary(y[split.val], tuned_val, ALERT_RATE))

# %% [markdown]
# ## Test window (reported once)

# %%
test_y = y[split.test]
baseline_test = baseline.predict_proba(X[split.test])[:, 1]
tuned_test = tuned.predict_proba(X[split.test])[:, 1]

results = {}
for label, probs in ((BASELINE_LABEL, baseline_test), (TUNED_LABEL, tuned_test)):
    summary = metric_summary(test_y, probs, ALERT_RATE)
    summary["brier"] = brier(test_y, probs)
    results[label] = summary
    print(label)
    print(summary)
    print(confusion_at_threshold(test_y, probs, threshold=0.5))

# %%
test_pr_auc = results[TUNED_LABEL]["pr_auc"]
if FD01_TEST_PR_AUC is None:
    print("FD01 test PR-AUC not recorded yet; the decision needs that value.")
else:
    print(f"tuned XGBoost minus FD01 logistic test PR-AUC: {test_pr_auc - FD01_TEST_PR_AUC:+.4f}")

# %% [markdown]
# ## Figures

# %%
title_suffix = f"{DATASET}, test split"
plots.plot_pr_curves(
    test_y,
    {"Baseline": baseline_test, "Tuned": tuned_test},
    f"PR curve, {title_suffix}, baseline vs tuned",
    FIG_DIR / "fd02_pr_curve.png",
)
importance = gain_importance(tuned)
plots.plot_gain_importance(
    importance,
    f"Feature importance by gain, {title_suffix}, tuned XGBoost",
    FIG_DIR / "fd02_feature_importance.png",
)
plots.plot_calibration(
    test_y,
    tuned_test,
    f"Reliability diagram, {title_suffix}, tuned XGBoost",
    FIG_DIR / "fd02_calibration.png",
)
importance.to_csv(REPORT_DIR / "fd02_feature_importance.csv", index=False)
print(importance.head(10))

# %% [markdown]
# ## Exports for FD05
# `fd02_scores.csv` holds validation and test rows only, scored by the tuned model.

# %%
DATA_DIR.mkdir(parents=True, exist_ok=True)
tuned.get_booster().save_model(str(DATA_DIR / "fd02_xgb_tuned.json"))
export = build_score_export(df, split, tuned)
write_score_export(export, DATA_DIR / "fd02_scores.csv")
print(export["split"].value_counts())
