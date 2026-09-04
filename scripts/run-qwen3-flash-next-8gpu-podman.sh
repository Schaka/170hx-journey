#!/bin/bash
# Thin wrapper around the "qwen8gpu" service in compose/docker-compose.yml.
# Tensor-parallel-size 4 + pipeline-parallel-size 2, spread across both PLX
# switches so all 8 GPUs are in use, tuned for many concurrent sessions
# (max-num-seqs 64) rather than single-stream latency. GPU power draw stays
# low (75-95W) and utilization high (85-90%) under this config even at low
# concurrency -- that is inter-GPU synchronization cost across PCIe Gen2,
# not idle time, and it needs real concurrent load to pay off. Single-stream
# work should use run-qwen3-flash-next-podman.sh (TP4, 4 GPUs) instead.
# dsv4-a100, qwen3-flash-next, and qwen3-flash-next-8gpu are mutually
# exclusive on this host (all three bind :8098), so bring the others down
# first.

cd "$(dirname "${BASH_SOURCE[0]}")/../compose" || exit 1

podman compose --profile dsv4 down >/dev/null 2>&1
podman compose --profile qwen down >/dev/null 2>&1
podman compose --profile qwen8gpu up -d
echo "launched qwen3-flash-next-8gpu on :8098"
echo "watch: podman logs -f qwen3-flash-next-8gpu"
