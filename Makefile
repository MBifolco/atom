.PHONY: test coverage coverage-html coverage-report install-test-deps

install-test-deps:
	pip install -r requirements-test.txt

test:
	python -m pytest tests/

coverage:
	python -m pytest tests/ --cov=src --cov=training/src --cov-report=term-missing --cov-fail-under=80

coverage-html:
	python -m pytest tests/ --cov=src --cov=training/src --cov-report=html
	@echo "Coverage report generated in htmlcov/index.html"

coverage-report:
	coverage report --show-missing
