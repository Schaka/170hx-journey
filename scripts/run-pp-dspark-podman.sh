#!/bin/bash
# Thin wrapper around the "dsv4" service in compose/docker-compose.yml. GPU
# recovery logic runs first because a wedged GPU state (from a prior crash)
# fails silently inside a container otherwise.

cd "$(dirname "${BASH_SOURCE[0]}")/../compose" || exit 1

if ! podman run --rm --device nvidia.com/gpu=all \
        --entrypoint python3 "${DSV4_IMAGE:-dsv4-a100:devel}" \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(4)]' \
        >/dev/null 2>&1; then
  echo "GPUs wedged -> recovering"
  nvidia-smi -r -i 0,1,2,3 >/dev/null 2>&1
  sudo rmmod nvidia_uvm 2>/dev/null; sudo modprobe nvidia_uvm
  for g in 0 1 2 3; do sudo nvidia-smi -i "$g" -pl 180 >/dev/null; done
fi

podman compose --profile dsv4 up -d
echo "launched dsv4-a100 on :8098"
echo "watch: podman logs -f dsv4-a100"
