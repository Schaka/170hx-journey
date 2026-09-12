#!/usr/bin/env python3
"""Check that the server reads its own prompt correctly.

The prompt is a block of dense technical text that names one rare string.
The model is asked for that string alone. A wrong answer means the server
read the prompt incorrectly, because the string is right there in the
context. The failure is a garbled copy of the string, not a refusal.

Usage:

    python3 context_check.py [PORT] [TRIALS] [TOKENS]

Every trial puts a different salt at the front, so no trial reads the
answer out of the prefix cache. The script prints an M-of-N count.
"""
import json
import re
import sys
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "8098"
TRIALS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"

NEEDLE = "/home/Schaka/Documents/rocm-gfx803"
FACT = f'The session workspace is "{NEEDLE}". All paths are relative to it.'

# Dense prose, not repeated filler. Repeated filler does not reproduce the
# fault, because the model can rebuild it from any part of the context.
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
    """Dense prose with one rare fact at the end.

    Every line differs, so the model cannot rebuild a missing line from its
    neighbours, and the salt makes each trial miss the prefix cache.
    """
    lines = [f"Report {salt}. Read the notes, then answer the question.", ""]
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
    lines += ["", FACT]
    return "\n".join(lines)


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
    data = json.load(urllib.request.urlopen(req, timeout=1800))
    message = data["choices"][0]["message"]
    # A run that spends its whole budget thinking has no content. Read the
    # reasoning instead, because the path still shows there.
    answer = ((message.get("content") or "") or (message.get("reasoning") or ""))
    answer = answer.strip()
    usage = data["usage"]
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    return NEEDLE in answer, answer[-70:], usage["prompt_tokens"], cached


def main() -> int:
    wrong = []
    tokens = 0
    for trial in range(TRIALS):
        ok, answer, tokens, cached = once(trial + TOKENS)
        print(f"{trial:2d} ok={ok} cached={cached:>6}/{tokens:<6} {answer!r}")
        if not ok:
            wrong.append(answer)
    print(f"\nport {PORT}: {tokens} tokens, wrong {len(wrong)}/{TRIALS}")
    for answer in sorted(set(wrong)):
        print("   ", answer)
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
