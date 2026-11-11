.PHONY: help setup lint test docker-up docker-down docker-logs pull-model run-langgraph run-strands compare

TOPIC ?= "Artificial Intelligence"

help:
	@echo "Available commands:"
	@echo "  make setup            Install Python dependencies"
	@echo "  make lint             Run ruff linter"
	@echo "  make test             Run unit tests (no Docker needed)"
	@echo "  make docker-up        Start Ollama via Docker Compose"
	@echo "  make docker-down      Stop Ollama Docker container"
	@echo "  make docker-logs      View Ollama container logs"
	@echo "  make pull-model       Pull the configured local model into Ollama"
	@echo "  make run-langgraph    Run LangGraph pipeline (e.g. make run-langgraph TOPIC=\"Python\")"
	@echo "  make run-strands      Run Strands pipeline (e.g. make run-strands TOPIC=\"Python\")"
	@echo "  make compare          Run both side-by-side (e.g. make compare TOPIC=\"Python\")"

setup:
	pip install -e ".[dev]"

lint:
	python3 -m ruff check .

test:
	python3 -m unittest discover tests

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f ollama

pull-model:
	./scripts/pull-model.sh

run-langgraph:
	python3 -m langgraph_impl $(TOPIC)

run-strands:
	python3 -m strands_impl $(TOPIC)

compare:
	python3 compare.py $(TOPIC)

