#!/usr/bin/env bash
# Build a DeepSeek-V4.1-capable vllm-backport image for sm_80.
#
# Needs: a vLLM checkout with PR 56201 and upstream main fetched, and podman.
#   git clone --filter=blob:none https://github.com/vllm-project/vllm.git
#   cd vllm
#   git fetch https://github.com/vllm-project/vllm.git pull/56201/head:pr56201
#   git fetch https://github.com/vllm-project/vllm.git main:upstream-main
#
# BASE_REF must name upstream main, not a fork main. The diff below is the
# merge base against BASE_REF. A fork main gives a merge base thousands of
# commits back, and the diff then rewrites half the tree.
set -euo pipefail

BASE=${BASE:-docker.io/lazymio/vllm-backport:v0.12.0-sm80}
TAG=${TAG:-localhost/vllm-backport-v41:sm80}
VLLM_SRC=${VLLM_SRC:?set VLLM_SRC to the vLLM checkout}
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

# 2. Apply the upstream PR, then the anchor-based fixups for its rejects.
BASE_REF=${BASE_REF:-upstream-main}
MB=$( cd "$VLLM_SRC" && git merge-base "$BASE_REF" pr56201 )
NCOMMITS=$( cd "$VLLM_SRC" && git rev-list --count "$MB..pr56201" )
# The pull request holds fewer than 20 commits. A larger count means BASE_REF
# is not upstream main, and the diff would rewrite files the PR never touched.
if [ "$NCOMMITS" -gt 20 ]; then
  echo "BASE_REF=$BASE_REF gives $NCOMMITS commits from merge base $MB." >&2
  echo "Fetch upstream main and set BASE_REF to it." >&2
  exit 1
fi
echo "PR diff: $NCOMMITS commits from $MB"
( cd "$VLLM_SRC" && git diff "$MB" pr56201 -- vllm ) > "$WORK/v41.patch"
( cd "$WORK" && git apply --reject --whitespace=nowarn v41.patch || true )
# The fixups vendor files that the sm80 backport drops. Read them from the PR
# branch itself, not from whatever VLLM_SRC has checked out.
rm -rf "$WORK/prtree"
mkdir -p "$WORK/prtree"
( cd "$VLLM_SRC" && git archive pr56201 vllm ) | tar x -C "$WORK/prtree"
( cd "$WORK" && PR_TREE="$WORK/prtree" python3 "$KIT/fixups.py" )
find "$WORK/vllm" -name '*.rej' -delete

# 3. Drop in the sm_80 attention shim.
mkdir -p "$WORK/vllm/models/deepseek_v4_1/ampere"
cp "$KIT/ampere/__init__.py" "$KIT/ampere/ampere_sparse.py" \
   "$KIT/ampere/qnorm_rope_kv_insert.py" \
   "$WORK/vllm/models/deepseek_v4_1/ampere/"
# The relay is not Ampere-specific, so it sits beside the model itself.
cp "$KIT/ampere/pp_kv_group_relay.py" "$WORK/vllm/models/deepseek_v4_1/"

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
