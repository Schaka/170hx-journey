#!/bin/bash
# Thin wrapper around the "glm53flash" service in compose/docker-compose.yml.
# Runs GLM-5.3-Flash-AWQ-W4A16 (wtdcode/GLM-5.3-Flash-AWQ-W4A16) on the
# wtdcode/vllm-backport image. Uses pipeline-parallel-size 4, not
# tensor-parallel, for the same reason as the DeepSeek-V4 profiles: this
# hardware has no P2P over PCIe Gen2, and pipeline parallel moves far less
# data across that link. This profile is untested on this hardware as of
# this writing. Every model-serving profile binds :8098, so this stops the
# others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile glm53flash up -d
echo "launched glm-5.3-flash on :8098"
echo "watch: podman logs -f glm-5.3-flash"
