# Model serving

Every backend starts through
[compose/docker-compose.yml](compose/docker-compose.yml). The compose file defines
nine profiles: `dsv4`, `dsv4backport`, `qwen`, `qwen8gpu`, `qwenawq`, `glm53flash`,
`glm53flash8gpu`, `glm53int4`, and `glm53int48gpu`. All nine are mutually exclusive
on this host, because they all bind port 8098.

## Model files

Model weights live on the `/models` mount (`/dev/md0`), one directory per model:

| model | path |
|---|---|
| DeepSeek-V4-Flash-0731 | `/models/deepseek-ai/DeepSeek-V4-Flash-0731` |
| Qwen3.8-Flash-Next-FP8 | `/models/Qwen` |
| Qwen3.8-Flash-Next-AWQ-W4A16 | `/models/Qwen3.8-Flash-Next-AWQ-W4A16` |
| GLM-5.3-Flash-AWQ-W4A16 | `/models/GLM-5.3-Flash-AWQ-W4A16` |
| GLM-5.3-AWQ-INT4 | `/models/GLM-5.3-AWQ-INT4` |

The compose file mounts these paths by default. Set `DSV4_MODEL`, `QWEN_MODEL`,
`QWEN_AWQ_MODEL`, `GLM_FLASH_MODEL`, or `GLM_INT4_MODEL` to override the path
for a single run.

The `qwen`, `qwen8gpu`, and `qwenawq` services need a
`chat_template_lenient_system.jinja` file next to their model weights. This
file does not come from the model download. Right now `/models/Qwen` and
`/models/Qwen3.8-Flash-Next-AWQ-W4A16` hold only the stock
`chat_template.jinja`. These three services do not start until someone adds
the lenient template back to both directories.

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
on its own.

### Alternative backend: vllm-backport

The `dsv4backport` profile runs the same DeepSeek-V4-Flash-0731 model on the
upstream [wtdcode/vllm-backport](https://github.com/wtdcode/vllm-backport) image
(`lazymio/vllm-backport:latest-sm80`) instead of the fork above. Start it with
[`run-dsv4-backport-podman.sh`](scripts/run-dsv4-backport-podman.sh).

This profile sets `--pipeline-parallel-size 4`, not tensor-parallel, for the same
reason as the fork. This hardware has no P2P over PCIe Gen2, and pipeline parallel
moves far less data across that link. It sets `--kv-cache-dtype fp8_ds_mla`, the
value that vllm-backport requires for this model, in place of the fork's `fp8`.

The vllm-backport maintainer reports a fix for the reasoning-loop problem on the
`main` branch. See [issue #22](https://github.com/wtdcode/vllm-backport/issues/22)
for the report. This profile is a way to test that fix on this hardware. Patch
`0023` from the fork is not present in this backend, because vllm-backport is a
separate codebase. Compare the two profiles for reasoning-loop behavior before you
pick one for regular use.

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

### Qwen3.8-Flash-Next-AWQ-W4A16: an untested alternative quantization

The `qwenawq` profile runs
[wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16](https://huggingface.co/wtdcode/Qwen3.8-Flash-Next-AWQ-W4A16)
on the same `lazymio/vllm-backport:latest-sm80` image, with the routed experts
in INT4 and everything else in BF16. It uses `--tensor-parallel-size 4`, the
same as the FP8 `qwen` profile, because this is the same base architecture
already proven on this hardware. It sets `--compilation-config` to
`{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY"}`, the value the AWQ build needs
in place of `FULL_AND_PIECEWISE`. Nobody has run this profile on this
hardware yet. Test it before you rely on it for regular use.

### LMCache is not enabled for this model

The `lazymio/vllm-backport:latest-sm80` image bundles LMCache, a KV cache offload
project used by vllm-backport. The compose file does not turn this feature on for
Qwen3.8-Flash-Next. Setting `--kv-transfer-config` with `LMCacheConnectorV1` raises
`ValueError: Failed to promote local KV cache specs to one unified type` at engine
start, on this model. The model mixes attention layers with mamba or GDN layers. On this vLLM version, that
mix does not unify into one KV cache spec under this connector.

## GLM-5.3

Four profiles run GLM-5.3 on the `lazymio/vllm-backport:latest-sm80` image, two
per quantization: one on 4 GPUs and one on all 8.

### glm53flash: tensor-parallel on 4 GPUs

`glm53flash` runs
[wtdcode/GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16)
with `--tensor-parallel-size 4`, matching wtdcode's own recipe, tested working
on this hardware. Start it with
[`run-glm53-flash-podman.sh`](scripts/run-glm53-flash-podman.sh).

An earlier version of this profile used `--pipeline-parallel-size 4` instead,
for the same PCIe Gen2 reason as the DeepSeek-V4 profiles. That did not work:
GLM-5.3's roughly 176 GB of weights do not split evenly across pipeline
stages. One stage overflowed a 64 GB card during warmup, before the engine
even started sizing the KV cache. Tensor parallel splits every layer's
weights evenly by construction, so it does not have this problem. It also
matches the recipe wtdcode already tested for this model.

Tensor parallel fixed the overflow, but this profile then failed to fit a
full 1,000,000-token context. `vllm`'s own log measured the KV cache math
directly at each step:

| change | available KV cache | estimated max context |
|---|---|---|
| baseline (TP4, `--gpu-memory-utilization 0.85`, `--max-num-batched-tokens 8192`) | 1.08 GiB | 71,424 |
| `--max-num-batched-tokens` down to 2048 | 1.88 GiB | 140,544 |
| add `--language-model-only` (drops the unused vision/video encoder cache) | 2.46 GiB | 188,928 |
| `--gpu-memory-utilization` up to 0.90 | 5.63 GiB | 457,344 |

A single 1,000,000-token request needs 12.04 GiB of KV cache on this model,
so none of these changes reach it on 4 GPUs. `--max-model-len 500000`
(`GLM_MAXLEN`) does fit, and this is the value the profile uses today.
`--max-num-batched-tokens` and `cudagraph_capture_sizes` were only tested at
the low end (2048 and `[1,2,4,8]`). Higher values, closer to wtdcode's
original recipe, can fit within the memory this configuration freed up.
Nobody tested that combination yet.

Measured single-stream decode on this profile reaches about 47 tokens per
second, close to 3 times faster than `glm53flash8gpu` below. MTP speculative
decoding also works far better here: a 45% draft acceptance rate and a mean
acceptance length of 2.35, against `glm53flash8gpu`'s 4 to 5% and 1.12. Use
`glm53flash` over `glm53flash8gpu` unless a request needs more than 500,000
tokens of context.

### glm53flash8gpu: pipeline-parallel on 8 GPUs

`glm53flash8gpu` runs the same model with `--pipeline-parallel-size 8`
instead, spreading the same weights across all 8 GPUs. Tested working on this
hardware: peak memory per card drops to about 52 GB out of 64 GB. That leaves enough room for `--max-model-len 1000000` (`GLM_MAXLEN`). That
value stays close to the model's native ceiling of `1048576`, and it matches
the limit set on every other model in this file. Start it with
[`run-glm53-flash-8gpu-podman.sh`](scripts/run-glm53-flash-8gpu-podman.sh).

Single-stream decode measures 13 to 16 tokens per second on this profile,
well below `glm53flash`. Every token crosses 7 inter-GPU handoffs, over a
Gen2 x4 link with no P2P, instead of the 3 handoffs that
`--pipeline-parallel-size 4` needs. `glm53flash8gpu` pays a larger version of
the same hop cost documented for `qwen8gpu` below. The MTP speculative
decoding in this profile barely helps. Mean acceptance length measures
around 1.12, with a 4 to 5% draft acceptance rate. Most draft tokens go to
waste, and each one still pays for a full round trip through all 8 stages.
`glm53flash` above wins on speed at every context length it can reach. When a
request needs more than 500,000 tokens of context, use this profile instead.

### glm53int4 and glm53int48gpu: the same model, a different quantization

`glm53int4` and `glm53int48gpu` run
[cyankiwi/GLM-5.3-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-AWQ-INT4),
a different quantization of the same base model, published outside the
vllm-backport project. vllm-backport's support for this specific
quantization is unconfirmed either way. `glm53int4` still uses
`--pipeline-parallel-size 4`. Nobody applied the `glm53flash` fix above to it
yet: tensor parallel, `--language-model-only`, a smaller
`--max-num-batched-tokens`, and a higher `--gpu-memory-utilization`. It
likely fails the same way `glm53flash` originally did. `glm53int48gpu` uses
`--pipeline-parallel-size 8` and mirrors `glm53flash8gpu`'s working
configuration, but nobody ran it on this hardware yet. Start them with
[`run-glm53-int4-podman.sh`](scripts/run-glm53-int4-podman.sh) or
[`run-glm53-int4-8gpu-podman.sh`](scripts/run-glm53-int4-8gpu-podman.sh).

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
[`stop-all-podman.sh`](scripts/stop-all-podman.sh) holds the list of every
profile and a `stop_all` function that brings all of them down. Every other
script sources it and calls `stop_all` before it starts its own profile,
because all nine profiles share port 8098.

- `run-pp-dspark-podman.sh` starts the DeepSeek-V4-Flash `dsv4` service, the fork
  build. It also checks GPU health before the start. If it finds a wedged GPU
  state, it recovers that state.
- `run-dsv4-backport-podman.sh` starts the DeepSeek-V4-Flash `dsv4backport`
  service, the vllm-backport build. It checks GPU health the same way as
  `run-pp-dspark-podman.sh`.
- `run-qwen3-flash-next-podman.sh` starts the Qwen3.8-Flash-Next `qwen` profile
  (single-stream, FP8, 4 GPUs).
- `run-qwen3-flash-next-8gpu-podman.sh` starts the Qwen3.8-Flash-Next `qwen8gpu`
  profile (many concurrent sessions, FP8, all 8 GPUs).
- `run-qwen3-flash-next-awq-podman.sh` starts the Qwen3.8-Flash-Next `qwenawq`
  profile (AWQ W4A16 quantization).
- `run-glm53-flash-podman.sh` starts the `glm53flash` profile
  (GLM-5.3-Flash-AWQ-W4A16, 4 GPUs). Does not start on this hardware today.
- `run-glm53-flash-8gpu-podman.sh` starts the `glm53flash8gpu` profile
  (GLM-5.3-Flash-AWQ-W4A16, all 8 GPUs).
- `run-glm53-int4-podman.sh` starts the `glm53int4` profile
  (GLM-5.3-AWQ-INT4, 4 GPUs). Does not start on this hardware today.
- `run-glm53-int4-8gpu-podman.sh` starts the `glm53int48gpu` profile
  (GLM-5.3-AWQ-INT4, all 8 GPUs).
