#!/bin/bash
# Thin wrapper around the "qwen" service in compose/docker-compose.yml.
# dsv4-a100 and qwen3-flash-next are mutually exclusive on this host (both
# bind :8098), so bring the other one down first.

cd "$(dirname "${BASH_SOURCE[0]}")/../compose" || exit 1

podman compose --profile dsv4 down >/dev/null 2>&1
podman compose --profile qwen up -d
echo "launched qwen3-flash-next on :8098"
echo "watch: podman logs -f qwen3-flash-next"
