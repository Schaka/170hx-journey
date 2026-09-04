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

### Two profiles: single-stream vs. many concurrent sessions

The `qwen` profile above uses tensor-parallel size 4 on 4 GPUs, tuned for
single-stream decode speed. A second profile, `qwen8gpu`, adds
`--pipeline-parallel-size 2` on top, so the model splits across both PLX switch
groups and uses all 8 GPUs. This trades single-stream latency for higher
concurrency headroom: `max-num-seqs` goes from 8 to 64.

Pipeline parallelism only pays off with several requests in flight at once, enough
to keep every pipeline stage busy. A single request through `qwen8gpu` crosses an
extra hop between the two switch groups. That hop carries no throughput benefit on
its own, and the request measures the same speed as `qwen`, or slower.

`nvidia-smi`'s `utilization.gpu` field reads 85 to 90% under this profile, even at
low concurrency, but power draw stays at 75 to 95W per card. That combination means
the GPUs spend most of their time on inter-GPU synchronization over PCIe Gen2, not
on compute. Serve many concurrent agent sessions at once, and `qwen8gpu` is the
right profile. Serve one interactive session, and `qwen` is the right profile.

Measured aggregate completion throughput on `qwen8gpu`, 300-token completions:

| concurrent requests | aggregate tok/s |
|---|---|
| 16 | 128.7 |
| 32 | 193.4 |
| 64 (the `max-num-seqs` ceiling) | 196.0 |

Throughput plateaus between 32 and 64 concurrent requests. Pushing concurrency past
32 buys almost no more aggregate throughput. It only splits the same total across
more sessions, so each one gets a smaller share.

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
- `run-qwen3-flash-next-podman.sh` starts the Qwen3.8-Flash-Next `qwen` profile
  (single-stream, 4 GPUs). It stops the DeepSeek-V4 service first, because the two
  share port 8098.
- `run-qwen3-flash-next-8gpu-podman.sh` starts the Qwen3.8-Flash-Next `qwen8gpu`
  profile (many concurrent sessions, all 8 GPUs). It stops both other services
  first, for the same reason.
