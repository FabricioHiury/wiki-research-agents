#!/usr/bin/env bash
set -euo pipefail

# Load environment variables from .env if present
if [ -f .env ]; then
  # Export non-comment lines
  export $(grep -v '^#' .env | grep -v '^$' | xargs)
fi

MODEL="${1:-${MODEL_NAME:-llama3.1:8b}}"
HOST="${OLLAMA_HOST:-http://localhost:11434}"

echo "=================================================="
echo "Pulling local model '${MODEL}' via Ollama"
echo "Ollama host: ${HOST}"
echo "=================================================="

# Check if Ollama container is running
if docker compose ps --services --filter "status=running" 2>/dev/null | grep -q "ollama"; then
  echo "Found running 'ollama' docker container. Executing pull inside container..."
  docker compose exec -T ollama ollama pull "${MODEL}"
elif curl -s -f "${HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama is reachable at ${HOST}. Triggering pull API..."
  curl -X POST "${HOST}/api/pull" -d "{\"name\": \"${MODEL}\"}"
else
  echo "Error: Ollama is not running at ${HOST} and docker container 'ollama' is not up."
  echo "Please start the Ollama service first:"
  echo "  docker compose up -d"
  echo "Then run this script again."
  exit 1
fi

echo ""
echo "Model '${MODEL}' is ready!"
