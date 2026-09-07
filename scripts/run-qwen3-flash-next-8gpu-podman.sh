#!/bin/bash
# Thin wrapper around the "qwen8gpu" service in compose/docker-compose.yml.
# Tensor-parallel-size 4 + pipeline-parallel-size 2, spread across both PLX
# switches so all 8 GPUs are in use, tuned for many concurrent sessions
# (max-num-seqs 64) rather than single-stream latency. GPU power draw stays
# low (75-95W) and utilization high (85-90%) under this config even at low
# concurrency -- that is inter-GPU synchronization cost across PCIe Gen2,
# not idle time, and it needs real concurrent load to pay off. Single-stream
# work should use run-qwen3-flash-next-podman.sh (TP4, 4 GPUs) instead. Every
# model-serving profile binds :8098, so this stops the others first.

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "$SCRIPT_DIR/stop-all-podman.sh"

stop_all
cd "$SCRIPT_DIR/../compose" || exit 1
podman compose --profile qwen8gpu up -d
echo "launched qwen3-flash-next-8gpu on :8098"
echo "watch: podman logs -f qwen3-flash-next-8gpu"
