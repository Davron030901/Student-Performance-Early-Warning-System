.PHONY: install data train test api docker-build docker-up clean

install:
	pip install -r requirements.txt

data:            ## generate the synthetic OULAD-schema dataset into data/raw/
	python -m src.data.generate_synthetic_oulad

train:           ## run the full pipeline: features -> baselines -> model -> reports -> artifact
	python -m src.models.train
	$(MAKE) cohort

cohort:          ## score a real held-out cohort into models/artifacts/demo_cohort.json (served to the dashboard)
	python -m src.data.build_demo_cohort

test:
	python -m pytest tests/ -v

api:             ## serve the model locally on :8000
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker compose build

docker-up:
	docker compose up

clean:
	rm -rf __pycache__ .pytest_cache reports/*.png reports/*.csv
