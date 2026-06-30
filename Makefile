.PHONY: all install lint format test test_integration typing

all: lint typing test

install:
	uv sync --all-groups

test:
	uv run --group test pytest tests/unit_tests

test_integration:
	uv run --group test --group test_integration pytest tests/integration_tests

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

typing:
	uv run --group typing mypy langchain_engram
