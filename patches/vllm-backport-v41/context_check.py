#!/usr/bin/env python3
"""Check that the server reads a fact from deep inside its own prompt.

The prompt names one rare path once, at a chosen depth, and the question asks
for that path alone. A wrong answer means the server did not reach the fact,
because the fact is right there in the context. The failure is a garbled copy
of the path or a run of unrelated lines, not a refusal.

Depth is the point of the script. V4.1 attends to a local window plus the
blocks its sparse indexer picks. A fact near the end of the prompt sits in the
local window, so the model reads it without the indexer and the run passes on
a broken indexer. Plant the fact in the first few percent instead, and the
answer depends on the indexer picking the right blocks.

Usage:

    python3 context_check.py [PORT] [TRIALS] [TOKENS] [DEPTH_PERCENT]

Every trial carries a different salt, so no trial reads the answer out of the
prefix cache.
"""
import json
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8098"
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 10
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 40000
DEPTH = int(sys.argv[4]) if len(sys.argv) > 4 else 5
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"

NEEDLE = "/home/Schaka/Documents/rocm-gfx803"
FACT = f'The session workspace is "{NEEDLE}". All paths are relative to it.'

# Dense prose, not repeated filler. Repeated filler does not reproduce a fault,
# because the model can rebuild it from any part of the context.
SUBJECTS = [
    "scheduler", "allocator", "indexer", "compressor", "sampler", "drafter",
    "tokenizer", "profiler", "collector", "dispatcher", "planner", "validator",
]
OBJECTS = [
    "block table", "page size", "slot mapping", "score matrix", "draft budget",
    "rotary cache", "weight scale", "residual stream", "token stream",
    "hash table", "state cache", "query group",
]
VERBS = [
    "admits", "rejects", "pads", "splits", "merges", "caps", "reorders",
    "quantizes", "gathers", "publishes", "retires", "clamps",
]


def build(salt: int) -> str:
    """Dense prose with one rare fact planted at `DEPTH` percent.

    Every line differs, so the model cannot rebuild the fact from its
    neighbours, and the salt makes each trial miss the prefix cache.
    """
    lines = []
    i = 0
    while len(" ".join(lines)) < TOKENS * 4:
        subject = SUBJECTS[(i * 7 + salt) % len(SUBJECTS)]
        verb = VERBS[(i * 5 + salt) % len(VERBS)]
        obj = OBJECTS[(i * 3 + salt) % len(OBJECTS)]
        lines.append(
            f"Note {salt}-{i:04d}: the {subject} {verb} the {obj} "
            f"after step {(i * 13 + salt) % 991}."
        )
        i += 1
    lines.insert(max(1, len(lines) * DEPTH // 100), FACT)
    head = [f"Report {salt}. Read the notes, then answer the question.", ""]
    return "\n".join(head + lines)


def once(salt: int):
    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "system", "content": "You answer with facts from the prompt."},
            {"role": "user", "content": build(salt)},
            {"role": "user", "content": "What is the session workspace? Answer with the path alone."},
        ],
        "max_tokens": 3000,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        BASE, json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    data = json.load(urllib.request.urlopen(req, timeout=3600))
    message = data["choices"][0]["message"]
    # A run that spends its whole budget thinking has no content. Read the
    # reasoning instead, because the path still shows there.
    answer = ((message.get("content") or "") or (message.get("reasoning") or ""))
    answer = answer.strip()
    return NEEDLE in answer, answer[-80:], data["usage"]["prompt_tokens"]


def main() -> int:
    wrong = []
    tokens = 0
    for trial in range(TRIALS):
        ok, answer, tokens = once(trial * 97 + DEPTH)
        print(f"{trial:2d} ok={ok} {answer!r}", flush=True)
        if not ok:
            wrong.append(answer)
    print(f"\nport {PORT} depth {DEPTH}%: {tokens} tokens, wrong {len(wrong)}/{TRIALS}")
    for answer in sorted(set(wrong)):
        print("   ", answer)
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
