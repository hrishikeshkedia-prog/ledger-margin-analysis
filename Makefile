.PHONY: synth clean-data test dashboard

synth:
	python -m src.synth --rows 20000 --seed 42 --out data/synthetic/ledger.csv

clean-data:
	python -m src.pipeline --in data/synthetic/ledger.csv --overhead-in data/synthetic/overhead.csv --out-dir data/processed

test:
	python -m pytest

dashboard:
	streamlit run dashboard/app.py
