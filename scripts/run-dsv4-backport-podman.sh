#!/bin/bash
# Thin wrapper around the "dsv4backport" service in compose/docker-compose.yml.
# Runs DeepSeek-V4-Flash-0731 on the upstream wtdcode/vllm-backport image instead
# of the Schaka/deepseek-v4-cmp170hx fork. Uses pipeline-parallel-size 4, not
# tensor-parallel, for the same reason as the fork: this hardware has no P2P
# over PCIe Gen2, and pipeline parallel moves far less data across that link.
# GPU recovery logic runs first because a wedged GPU state (from a prior crash)
# fails silently inside a container otherwise. dsv4-a100, dsv4-backport,
# qwen3-flash-next, and qwen3-flash-next-8gpu are mutually exclusive on this
# host (all four bind :8098), so bring the others down first.

cd "$(dirname "${BASH_SOURCE[0]}")/../compose" || exit 1

if ! podman run --rm --device nvidia.com/gpu=all \
        --entrypoint python3 "${DSV4_BACKPORT_IMAGE:-docker.io/lazymio/vllm-backport:latest-sm80}" \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(4)]' \
        >/dev/null 2>&1; then
  echo "GPUs wedged -> recovering"
  nvidia-smi -r -i 0,1,2,3 >/dev/null 2>&1
  sudo rmmod nvidia_uvm 2>/dev/null; sudo modprobe nvidia_uvm
  for g in 0 1 2 3; do sudo nvidia-smi -i "$g" -pl 180 >/dev/null; done
fi

podman compose --profile dsv4 down >/dev/null 2>&1
podman compose --profile qwen down >/dev/null 2>&1
podman compose --profile qwen8gpu down >/dev/null 2>&1
podman compose --profile dsv4backport up -d
echo "launched dsv4-backport on :8098"
echo "watch: podman logs -f dsv4-backport"
