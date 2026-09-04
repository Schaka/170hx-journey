#!/bin/bash
# Qwen3.8-Flash-Next-FP8 on 4x CMP 170HX, via wtdcode/vllm-backport's prebuilt
# lazymio/vllm-backport:latest-sm80 image (NOT this fork's own dsv4-a100:devel --
# separate image, separate stack). Same host, same port (8098) as the DeepSeek-V4
# server for a one-line switch in opencode.json (see opencode's "qwen3fn" model
# entry under the "dsv4" provider, same baseURL).
#
# Recipe is wtdcode's own tested config for this model
# (https://github.com/wtdcode/vllm-backport#qwen38-flash-next-v090), used as-is:
# TP=4 + --enable-expert-parallel, NOT this repo's usual PP4 -- EP is required to
# keep the 640-wide expert intermediate whole (160/rank under plain TP is not a
# multiple of the 128x128 fp8 block, forcing the SM80-incompatible Triton fp8 MoE
# path). PP+EP together is untested upstream; don't experiment with it on the
# first run.
#
# Sampling defaults come from Qwen's own official thinking-mode recommendation
# (temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0,
# repetition_penalty=1.0) -- set server-side via --override-generation-config so
# no client-side config is required, matching how this repo's DeepSeek launcher
# pins top_p (see SETTINGS.md).
#
# chat_template_lenient_system.jinja: the stock chat_template.jinja raises
# jinja2.exceptions.TemplateError('System message must be at the beginning.')
# for any role=="system" message not at index 0. DeepSeek's tokenizer has no
# such constraint (handles system/developer roles per-message anywhere), so
# switching an in-progress opencode session from dsv4 to this model hits it
# immediately if the accumulated history ever picked up a system-role message
# past index 0 (a mid-session reminder, a compaction re-injection, etc.) --
# confirmed live 2026-09-01. The lenient copy (same file, one block changed:
# see its diff against the stock template) renders a non-first system message
# as its own inline <|im_start|>system...<|im_end|> turn instead of raising.
# Model-swap-mid-session keeps working at the cost of not matching Qwen's own
# strict system-message-placement contract exactly.

IMG="${QWEN_IMAGE:-docker.io/lazymio/vllm-backport:latest-sm80}"
MODEL="${QWEN_MODEL:-$HOME/models/models/Qwen/Qwen3.8-Flash-Next-FP8}"
MAXLEN="${QWEN_MAXLEN:-1000000}"
CHAT_TEMPLATE="${QWEN_CHAT_TEMPLATE:-$MODEL/chat_template_lenient_system.jinja}"
NAME="qwen3-flash-next"

podman image inspect "$IMG" >/dev/null 2>&1 || { echo "ERROR: image $IMG not pulled"; exit 1; }
[[ -d "$MODEL" ]] || { echo "ERROR: model dir not found: $MODEL"; exit 1; }
[[ -f "$CHAT_TEMPLATE" ]] || { echo "ERROR: chat template not found: $CHAT_TEMPLATE"; exit 1; }

# Frees port 8098 -- this and dsv4-a100 are mutually exclusive on this host.
podman stop -t 60 dsv4-a100 >/dev/null 2>&1
podman rm dsv4-a100 >/dev/null 2>&1
podman stop -t 60 "$NAME" >/dev/null 2>&1
podman rm "$NAME" >/dev/null 2>&1

# shellcheck disable=SC2086
podman run -d --name "$NAME" --device nvidia.com/gpu=all \
  -e VLLM_PLE_CPU_OFFLOAD=1 -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_ALGO=Ring -e NCCL_PROTO=Simple \
  -v "$MODEL":/model \
  --shm-size=16g -p 8098:8000 \
  "$IMG" /model \
  --host 0.0.0.0 --port 8000 \
  --served-model-name qwen3.8-flash-next \
  --chat-template "/model/$(basename "$CHAT_TEMPLATE")" \
  --tensor-parallel-size 4 --enable-expert-parallel \
  --max-model-len "$MAXLEN" --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.9 --disable-custom-all-reduce \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}' \
  --hf-overrides '{"text_config": {"rope_parameters": {"mrope_interleaved": true, "mrope_section": [11, 11, 10], "rope_type": "yarn", "rope_theta": 10000000, "partial_rotary_factor": 0.25, "factor": 4.0, "original_max_position_embeddings": 262144}}}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
  --enable-prompt-tokens-details --enable-prefix-caching --mamba-cache-mode align \
  --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":0.0,"repetition_penalty":1.0}' \
  >/dev/null
echo "launched $NAME on :8098  (maxlen $MAXLEN, TP4+EP, mtp-3)"
echo "watch: podman logs -f $NAME"
