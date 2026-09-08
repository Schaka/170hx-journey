#!/bin/bash
# Thin wrapper around the "glm53flash6gpu" service in compose/docker-compose.yml.
# Runs GLM-5.3-Flash-AWQ-W4A16 on 6 of the 8 GPUs, with pipeline-parallel-size 3
# and tensor-parallel-size 2 per pipeline stage. This keeps each all-reduce
# group at 2 GPUs, which lets vLLM use its custom all-reduce kernel instead of
# NCCL. Every model-serving profile binds :8098, so this stops the others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile glm53flash6gpu up -d
echo "launched glm-5.3-flash-6gpu on :8098"
echo "watch: podman logs -f glm-5.3-flash-6gpu"
