#!/bin/bash
# Thin wrapper around the "qwen" service in compose/docker-compose.yml. Every
# model-serving profile binds :8098, so this stops the others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile qwen up -d
echo "launched qwen3-flash-next on :8098"
echo "watch: podman logs -f qwen3-flash-next"
