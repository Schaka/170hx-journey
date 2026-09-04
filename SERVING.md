# Model serving

## DeepSeek-V4-Flash-0731

Served from my own fork,
[Schaka/deepseek-v4-cmp170hx](https://github.com/Schaka/deepseek-v4-cmp170hx) — a
patch set and container build on top of
[haosdent/vllm@dsv4-flash-a100](https://github.com/haosdent/vllm/tree/dsv4-flash-a100),
purpose-built for this hardware (PCIe Gen2, pipeline parallel over tensor parallel,
context-length fixes, repetition-loop tripwires). Full settings and reasoning for
every flag: that repo's `SETTINGS.md`.

One patch in that fork is **mandatory**, not optional, for any agent harness:

- [`patches/0023-dual-emit-reasoning-content-alias.patch`](https://github.com/Schaka/deepseek-v4-cmp170hx/blob/main/patches/0023-dual-emit-reasoning-content-alias.patch) —
  without it, this vLLM fork only emits reasoning text under a field called
  `reasoning`. Every client that expects the common `reasoning_content` convention
  (opencode included) fails to recognize it as reasoning at all, and the whole
  `<think>` block gets treated as plain assistant content — which pollutes
  conversation history and reliably drives the model into repetition loops over a
  long agentic session. This patch dual-emits both field names so either convention
  works.

The other repetition-loop-recovery patches in that fork (the `0016`–`0022` series)
have **not** been confirmed to actually do anything beyond what patch `0023` alone
fixes — worth testing whether
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) already handles
loop recovery out of the box without needing them, next time there's a spare cycle
for it.

### Agent client config (opencode example)

Base URL `http://<host>:8098/v1`, model config needs `reasoning_effort` set
**explicitly on every request** — a session that never sends it can drift into a
self-reinforcing repetition loop over many turns:

```json
{
    "provider": {
        "dsv4": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "vLLM dsv4",
            "options": { "baseURL": "http://<host>:8098/v1" },
            "models": {
                "dsv4s": {
                    "name": "dsv4-a100",
                    "limit": { "context": 1000000, "output": 32000 },
                    "options": {
                        "chat_template_kwargs": {
                            "thinking": true,
                            "reasoning_effort": "max"
                        }
                    }
                }
            }
        }
    }
}
```

`reasoning_effort: "max"`, not `high` — a
[community-reported bug](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/39)
has `low`/`high` sometimes rendering an empty thinking prefix that accumulates in
history over long agentic sessions. `high` still wins on pure recall benchmarks if
agentic tool use isn't the workload.

## Qwen3.8-Flash-Next-FP8

Served from [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport)'s
prebuilt `lazymio/vllm-backport:latest-sm80` image — a separate stack from the
DeepSeek-V4 fork above, same host, same port (8098), mutually exclusive with the
DeepSeek-V4 container. Launch recipe (TP4 + expert-parallel, MTP speculative decoding,
sampling defaults) follows wtdcode's own tested config for this model as-is; see
[`scripts/run-qwen3-flash-next-podman.sh`](scripts/run-qwen3-flash-next-podman.sh) for
the exact flags in use here.

## Launcher scripts

The scripts actually deployed and run on the workstation, copied verbatim into
[scripts/](scripts/):

- `run-pp-dspark-podman.sh` — DeepSeek-V4-Flash, pipeline-parallel + DSpark
  speculative decoding
- `run-qwen3-flash-next-podman.sh` — Qwen3.8-Flash-Next, tensor-parallel + expert
  parallel + MTP speculative decoding
