#!/usr/bin/env bash
# Set the per-card power limit on all 8 CMP 170HX cards.
#
# The cards default to 250 W and draw short peaks above it. This bounds the
# draw, and therefore the heat and the load on the two PSUs and their cables.
# 200 W holds the same sustained SM clock as 250 W, at 1470 to 1485 MHz.
#
# Usage: sudo ./gpu-power-limit.sh [watts]      (default 200, hardware max 300)
set -euo pipefail

WATTS=${1:-175}

if [ "$(id -u)" -ne 0 ]; then
  echo "Run this with sudo: nvidia-smi -pl needs root." >&2
  exit 1
fi

count=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
for i in $(seq 0 $((count - 1))); do
  nvidia-smi -i "$i" -pl "$WATTS" > /dev/null
done

nvidia-smi --query-gpu=index,power.limit,power.default_limit --format=csv
