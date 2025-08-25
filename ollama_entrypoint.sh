#!/bin/bash
set -e

# Start Ollama in the background
ollama serve  &
SERVER_PID=$!

# Wait for server to come up
sleep 10

# Read models from env
IFS=',' read -ra MODELS <<< "$OLLAMA_MODELS"

echo "Checking Ollama models..."
for model in "${MODELS[@]}"; do
    model=$(echo $model | xargs)
    if ! ollama list | grep -q "$model"; then
        echo "Pulling model $model..."
        ollama pull "$model"
    else
        echo "Model $model already exists."
    fi
done
# Kill background server
kill $SERVER_PID
wait $SERVER_PID || true

# Start Ollama server
exec ollama serve
