#!/bin/bash
# Podman adaptation of run-pp-dspark.sh for 4x CMP 170HX.
# Patches 0002-0006 are baked into dsv4-a100:devel at build time -> DSV4_NO_MOUNT=1.
# --runtime=nvidia is replaced by CDI: --device nvidia.com/gpu=all.
# This is the CURRENT, actively-used launcher -- deployed straight to the
# workstation's ~/run-pp-dspark-podman.sh. run-pp-dspark.sh (docker/--runtime=nvidia,
# no tool-call-parser/reasoning-parser flags) predates the podman switch and the
# 0016-0021 DSML/repetition patches; treat this file as authoritative.

IMG="${DSV4_IMAGE:-dsv4-a100:devel}"
MODEL="${DSV4_MODEL:-$HOME/models/models/deepseek-ai/DeepSeek-V4-Flash-0731}"
MAXLEN="${DSV4_MAXLEN:-1000000}"
ROW_CHUNK="${DSV4_ROW_CHUNK:-64}"
# DSV4_RECOVERY_MODE selects 0022's tripwire-recovery strategy: "nudge"
# (default, 0021's freeform-text splice) or "reopen" (0022's template-correct
# turn-boundary splice). See SETTINGS.md's "Repetition-loop recovery" section
# and patches/0022-repetition-recovery-reopen-turn-mode.patch -- reopen is
# untested pending a real run past ~100k context, staged so it can be A/B'd
# with a single env var, no rebuild.
RECOVERY_MODE="${DSV4_RECOVERY_MODE:-nudge}"
SPEC='--speculative-config {"method":"dspark","num_speculative_tokens":5}'
NAME="dsv4-a100"

while [ $# -gt 0 ]; do
  case "$1" in
    --plain)  SPEC=""; shift ;;
    --maxlen) MAXLEN="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

podman image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR: image $IMG not built"; exit 1; }
[[ -d "$MODEL" ]] || { echo "ERROR: model dir not found: $MODEL"; exit 1; }

podman stop -t 60 "$NAME" >/dev/null 2>&1
podman rm "$NAME" >/dev/null 2>&1

if ! podman run --rm --device nvidia.com/gpu=all \
        --entrypoint python3 "$IMG" \
        -c 'import torch;[torch.randn(8,8,device=f"cuda:{i}") for i in range(4)]' \
        >/dev/null 2>&1; then
  echo "GPUs wedged -> recovering"
  nvidia-smi -r -i 0,1,2,3 >/dev/null 2>&1
  sudo rmmod nvidia_uvm 2>/dev/null; sudo modprobe nvidia_uvm
  for g in 0 1 2 3; do sudo nvidia-smi -i "$g" -pl 180 >/dev/null; done
fi

# shellcheck disable=SC2086
podman run -d --name "$NAME" --device nvidia.com/gpu=all \
  -e HF_HUB_OFFLINE=1 -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e DSV4_LOGITS_ROW_CHUNK="$ROW_CHUNK" \
  -e DSV4_REP_FUZZY_MIN_HITS=15 \
  -e DSV4_RECOVERY_MODE="$RECOVERY_MODE" \
  -v "$MODEL":/model \
  --shm-size=16g -p 8098:8000 \
  "$IMG" vllm serve /model --served-model-name dsv4s \
  --pipeline-parallel-size 4 --kv-cache-dtype fp8 --block-size 256 \
  --max-model-len "$MAXLEN" --max-num-batched-tokens 2048 --trust-remote-code \
  --gpu-memory-utilization 0.85 --max-num-seqs 8 \
  --no-enable-flashinfer-autotune --tokenizer-mode deepseek_v4 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --override-generation-config '{"top_p":0.95}' \
  $SPEC >/dev/null
echo "launched $NAME on :8098  (maxlen $MAXLEN, row_chunk $ROW_CHUNK, recovery: $RECOVERY_MODE, spec: ${SPEC:-none})"
echo "watch: podman logs -f $NAME"
