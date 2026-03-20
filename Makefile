.PHONY: install-test-deps test test-unit test-integration test-training test-e2e test-slow baseline-local coverage coverage-report docs-links clean

install-test-deps:
	pip install -r requirements.txt

test:
	python -m pytest tests/ --no-cov

test-unit:
	python -m pytest -m unit --no-cov

test-integration:
	python -m pytest -m "integration and not slow" --no-cov

test-training:
	python -m pytest tests/training -m training -s --no-cov

test-e2e:
	python -m pytest tests/e2e -m e2e -s --no-cov

test-slow:
	python -m pytest -m slow --no-cov

baseline-local:
	python scripts/training/run_local_baseline.py --mode curriculum --timesteps 10000 --seed 1337 --cores 1 --device cpu

coverage:
	python -m pytest tests/ --cov=src --cov-report=term-missing --cov-fail-under=80

coverage-report:
	coverage report --show-missing

docs-links:
	python scripts/ops/check_markdown_links.py

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	rm -rf .pytest_cache htmlcov .coverage .coverage.*
