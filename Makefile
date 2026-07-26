.PHONY: synth synth-trips clean-data clean-trips test dashboard

synth:
	python -m src.synth --rows 20000 --seed 42 --out data/synthetic/ledger.csv

synth-trips:
	python -m src.synth --mode trips --rows 400 --seed 42 --out data/synthetic/trips.csv

clean-data:
	python -m src.pipeline --in data/synthetic/ledger.csv --overhead-in data/synthetic/overhead.csv --out-dir data/processed

clean-trips:
	python -m src.pipeline --trips data/synthetic/trips.csv --distances data/reference/distances.csv --out-dir data/processed

test:
	python -m pytest

dashboard:
	streamlit run dashboard/app.py
