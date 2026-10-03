install:
	python -m pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check .
format:
	ruff format .
run:
	uvicorn api.main:app --reload
infra:
	docker compose up -d postgres redis
infra-down:
	docker compose down
