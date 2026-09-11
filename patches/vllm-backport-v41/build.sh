#!/usr/bin/env bash
# Build a DeepSeek-V4.1-Flash image for sm_80 from the vllm-backport base.
#
# Needs: podman. The base image already carries the V4.1 model and the Ampere
# sparse-MLA backend, so this script only adds the pipeline-parallel relay,
# the DSpark embedding table and the reasoning_content alias.
set -euo pipefail

BASE=${BASE:-docker.io/lazymio/vllm-backport:v0.13.0-sm80}
TAG=${TAG:-localhost/vllm-backport-v41:sm80}
WORK=${WORK:-$PWD/work}
KIT=$(cd "$(dirname "$0")" && pwd)

# 1. Take the base image's vllm tree.
rm -rf "$WORK"
mkdir -p "$WORK"
cid=$(podman create --entrypoint /bin/sh "$BASE")
podman cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm" "$WORK/vllm"
podman rm "$cid" >/dev/null
find "$WORK/vllm" -name '*.pyc' -delete
find "$WORK/vllm" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
cp -r "$WORK/vllm" "$WORK/vllm.orig"

# 2. Apply the anchor-based edits.
( cd "$WORK" && python3 "$KIT/fixups.py" )

# 3. Drop in the relay module the edits import.
cp "$KIT/pp_kv_group_relay.py" "$WORK/vllm/models/deepseek_v4_1/"

# 4. Ship only the files that changed, found with a throwaway git index.
rm -rf "$WORK/ctx" "$WORK/gitdiff"
mkdir -p "$WORK/ctx" "$WORK/gitdiff"
cp -r "$WORK/vllm.orig" "$WORK/gitdiff/vllm"
( cd "$WORK/gitdiff" && git init -q . && git add -A >/dev/null 2>&1 \
    && git -c user.email=b@b -c user.name=b commit -qm base >/dev/null )
rm -rf "$WORK/gitdiff/vllm"
cp -r "$WORK/vllm" "$WORK/gitdiff/vllm"
( cd "$WORK/gitdiff" && git add -A >/dev/null 2>&1 \
    && git diff --cached --name-only | grep '^vllm/' > "$WORK/changed.txt" )
echo "shipping $(wc -l < "$WORK/changed.txt") files"
( cd "$WORK" && tar cf - -T changed.txt ) | tar xf - -C "$WORK/ctx"
cp "$KIT/Containerfile" "$WORK/ctx/"
podman build -t "$TAG" "$WORK/ctx"
echo "built $TAG"
