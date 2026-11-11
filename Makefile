.PHONY: install test notebook reproduce

install:
	pip install -e ".[dev]"

test:
	pytest

notebook:
	jupytext --to ipynb notebooks/fd02_cost_sensitive_xgboost.py
	jupyter nbconvert --to notebook --execute --inplace notebooks/fd02_cost_sensitive_xgboost.ipynb

reproduce: test notebook
