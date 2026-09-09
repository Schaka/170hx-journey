#!/bin/bash
# Thin wrapper around the "glm53int48gpu" service in compose/docker-compose.yml.
# Runs GLM-5.3-AWQ-INT4 with pipeline-parallel-size 8 across all 8 GPUs,
# instead of the 4 GPUs the "glm53int4" profile uses. GLM-5.3's weights split
# unevenly across pipeline stages, and on 4 GPUs one stage overflows a 64 GB
# card during warmup. Every model-serving profile
# binds :8098, so this stops the others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile glm53int48gpu up -d
echo "launched glm-5.3-int4-8gpu on :8098"
echo "watch: podman logs -f glm-5.3-int4-8gpu"
