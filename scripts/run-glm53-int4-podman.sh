#!/bin/bash
# Thin wrapper around the "glm53int4" service in compose/docker-compose.yml.
# Runs GLM-5.3-AWQ-INT4 (cyankiwi/GLM-5.3-AWQ-INT4) on the wtdcode/vllm-backport
# image, with the same launch settings as the glm53flash profile. This is a
# different quantization of the same base model, published outside the
# wtdcode/vllm-backport project, so vllm-backport's support for it is
# unconfirmed. This profile is untested on this hardware as of this writing.
# Every model-serving profile binds :8098, so this stops the others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile glm53int4 up -d
echo "launched glm-5.3-int4 on :8098"
echo "watch: podman logs -f glm-5.3-int4"
