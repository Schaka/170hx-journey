# Model serving

Both models start through [compose/docker-compose.yml](compose/docker-compose.yml).
The compose file defines two services, `dsv4` and `qwen`, as separate profiles. The
two services are mutually exclusive on this host, because both bind port 8098.

## DeepSeek-V4-Flash-0731

This server runs a fork:
[Schaka/deepseek-v4-cmp170hx](https://github.com/Schaka/deepseek-v4-cmp170hx). The
fork is a patch set and a container build. It sits on top of
[haosdent/vllm@dsv4-flash-a100](https://github.com/haosdent/vllm/tree/dsv4-flash-a100).
The fork targets this hardware directly: PCIe Gen2 support, pipeline parallel instead
of tensor parallel, context-length fixes, and repetition-loop tripwires. That repo's
`SETTINGS.md` file has the full setting list and the reason for each value.

One patch in that fork is required for any agent tool, not optional:

- [`patches/0023-dual-emit-reasoning-content-alias.patch`](https://github.com/Schaka/deepseek-v4-cmp170hx/blob/main/patches/0023-dual-emit-reasoning-content-alias.patch).
  Without this patch, this vLLM fork emits reasoning text under one field only, named
  `reasoning`. Most clients expect the common `reasoning_content` field name, opencode
  included. Without the patch, a client treats the whole `<think>` block as plain
  assistant content instead of reasoning. This error fills the conversation history
  with reasoning text. Over a long agentic session, this error reliably drives the
  model into a repetition loop. This patch emits both field names, so either
  convention works.

The fork also has a series of repetition-loop-recovery patches, numbered `0016`
through `0022`. These patches have no proven effect beyond what patch `0023` fixes
on its own. A future task is to test whether
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) already handles loop
recovery without these patches.

### Agent client configuration (opencode example)

Point the client at `http://<host>:8098/v1`. Set `reasoning_effort` on every request,
not just the first request. A session that never sends this value can drift into a
repetition loop over many turns, because the loop reinforces itself.

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

Set `reasoning_effort` to `max`, not `high`. A
[community-reported bug](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/discussions/39)
causes `low` and `high` to sometimes render an empty thinking prefix. This empty
prefix accumulates in the conversation history over a long agentic session.

When the task has no agentic tool use, `high` still wins on pure recall benchmarks.

## Qwen3.8-Flash-Next-FP8

This server runs a prebuilt image from
[wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport):
`lazymio/vllm-backport:latest-sm80`. This stack is separate from the DeepSeek-V4 fork
above. It shares the same host and the same port, 8098, so only one of the two
services runs at a time. The launch recipe uses tensor-parallel size 4, expert
parallel, and MTP speculative decoding. This recipe matches wtdcode's own tested
configuration for this model.

### LMCache is not enabled for this model

The `lazymio/vllm-backport:latest-sm80` image bundles LMCache, a KV cache offload
project used by vllm-backport. The compose file does not turn this feature on for
Qwen3.8-Flash-Next. Setting `--kv-transfer-config` with `LMCacheConnectorV1` raises
`ValueError: Failed to promote local KV cache specs to one unified type` at engine
start, on this model. The model mixes attention layers with mamba or GDN layers. On this vLLM version, that
mix does not unify into one KV cache spec under this connector.

## Persisted JIT and compile caches

Both vLLM containers write several just-in-time compile caches under `/root` inside
the container: `.cache/vllm/torch_compile_cache`, `.cache/flashinfer`, `.triton`,
`.nv/ComputeCache`, `.humming`, `.cupy`, and `.tilelang`. Without a host mount, each
container restart loses these caches. A lost cache turns every first start into a
multi-minute recompile and JIT-warmup pass. The compose file mounts each of these
paths to a directory under `cache/<service>/` on the host, so the caches survive a
container restart.

## Launcher scripts

The scripts in [scripts/](scripts/) run on the workstation. Each script is a thin
wrapper around one `podman compose --profile <name> up -d` call.

- `run-pp-dspark-podman.sh` starts the DeepSeek-V4-Flash service. It also checks GPU
  health before the start. If it finds a wedged GPU state, it recovers that state.
- `run-qwen3-flash-next-podman.sh` starts the Qwen3.8-Flash-Next service. It stops the
  DeepSeek-V4 service first, because the two share port 8098.
