#!/bin/bash
# Thin wrapper around the "qwenawq" service in compose/docker-compose.yml. Runs
# the AWQ W4A16 quantization of Qwen3.8-Flash-Next
# (wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16) instead of the FP8 build the "qwen"
# and "qwen8gpu" profiles run. Uses tensor-parallel-size 4, the same as the
# FP8 build, because this is the same base architecture already proven on
# this hardware. This profile is untested on this hardware as of this
# writing. Every model-serving profile binds :8098, so this stops the others
# first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile qwenawq up -d
echo "launched qwen3.8-flash-next-awq on :8098"
echo "watch: podman logs -f qwen3.8-flash-next-awq"
