# Data

Raw data is not committed to this repository. A local copy was downloaded on 2026-11-12 to `~/.local/share/fraud-data/creditcard.csv` (outside the repository, owner-approved).

| Field | Value |
|---|---|
| Kaggle slug | `mlg-ulb/creditcardfraud` |
| Kaggle reference | https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud |
| License (per Kaggle API, 2026-11-12) | DbCL-1.0 (Database Contents License) |
| Description | Real anonymized card transactions, September 2013 (284,807 rows, 492 frauds, PCA features V1-V28) |
| Notebook read path | `/kaggle/input/creditcardfraud/creditcard.csv` (attach the dataset in a Kaggle Notebook) |
| Local override | Set `FRAUD_DATA_PATH` to a local copy, for development only |
| Retrieval date | 2026-11-12, local copy with the Kaggle API (outside the repository) |

The `Time` column is seconds from the first transaction, so the full dataset covers about two days.

## Files written by FD02 (git-ignored, not committed)

| File | Written by | Used by |
|---|---|---|
| `fd02_scores.csv` | FD02 notebook (Kaggle run) | FD05, copy into the FD05 repo's `data/` folder |
| `fd02_xgb_tuned.json` | FD02 notebook (Kaggle run) | Reuse of the tuned booster |

`fd02_scores.csv` columns: `split` (`val` or `test`), `Time`, `Amount`, `Class`, `score` (predicted fraud probability of the tuned model). It holds validation and test rows only, never training rows.
