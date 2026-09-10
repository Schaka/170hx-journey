#!/bin/bash
# Thin wrapper around the "glm53mix8gpu" service in compose/docker-compose.yml.
# Runs the full GLM-5.3 from the INT4/INT8 mixed quantization at the model's
# full 1,048,576-token context, with pipeline-parallel-size 8 and MTP
# speculative decoding. Every model-serving profile binds :8098, so this stops
# the others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile glm53mix8gpu up -d
echo "launched glm-5.3-mix-8gpu on :8098"
echo "watch: podman logs -f glm-5.3-mix-8gpu"
