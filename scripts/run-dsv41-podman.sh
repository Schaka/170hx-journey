#!/bin/bash
# Thin wrapper around the "dsv41" service in compose/docker-compose.yml.
# Runs DeepSeek-V4.1-Flash on all 8 GPUs at the model's full 1,048,576-token
# context, with tensor-parallel-size 4, pipeline-parallel-size 2, and DSpark
# speculative decoding. Every model-serving profile binds :8098, so this stops
# the others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile dsv41 up -d
echo "launched deepseek-v4.1-flash on :8098"
echo "watch: podman logs -f deepseek-v41-flash"
