# Model serving

Every backend starts through
[compose/docker-compose.yml](compose/docker-compose.yml). The compose file defines
ten profiles: `dsv4`, `dsv4backport`, `qwen`, `qwen8gpu`, `qwenawq`, `glm53flash`,
`glm53flash8gpu`, `glm53int4`, `glm53int48gpu`, and `glm53mix8gpu`. All ten are
mutually exclusive on this host, because they all bind port 8098.

## Model files

Model weights live on the `/models` mount (`/dev/md0`), one directory per model:

| model | path |
|---|---|
| DeepSeek-V4-Flash-0731 | `/models/deepseek-ai/DeepSeek-V4-Flash-0731` |
| Qwen3.8-Flash-Next-FP8 | `/models/Qwen` |
| Qwen3.8-Flash-Next-AWQ-W4A16 | `/models/Qwen3.8-Flash-Next-AWQ-W4A16` |
| GLM-5.3-Flash-AWQ-W4A16 | `/models/GLM-5.3-Flash-AWQ-W4A16` |
| GLM-5.3 (full model, INT4 quant) | `/models/GLM-5.3-AWQ-INT4` |
| GLM-5.3 (full model, INT4/INT8 mixed quant) | `/models/GLM-5.3-Int4-Int8Mix` |

The compose file mounts these paths by default. Set `DSV4_MODEL`, `QWEN_MODEL`,
`QWEN_AWQ_MODEL`, `GLM_FLASH_MODEL`, `GLM_INT4_MODEL`, or `GLM_MIX_MODEL` to
override the path for a single run.

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

Five profiles run GLM-5.3 on the `lazymio/vllm-backport:latest-sm80` image.
The Flash quantization has three profiles, on 4, 6, and 8 GPUs. The INT4
quantization has two, on 4 and 8 GPUs.

### A patch for MTP speculative decoding under pipeline parallel

`glm53flash6gpu` and `glm53flash8gpu` below both use
pipeline parallel, and both mount a patch file over the vllm-backport image:
[`patches/vllm-backport/mtp-embed-from-checkpoint.py`](patches/vllm-backport/mtp-embed-from-checkpoint.py),
at
`/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu/spec_decode/eagle/utils.py`.

Under pipeline parallel, vLLM's MTP drafter runs on the last stage, but the
target model's token embedding lives on the first stage. The drafter needs
its own copy, and the vllm-backport image on this hardware has no code path
that loads one for it. The draft then runs on uninitialized GPU memory. It
still emits tokens, so the server starts and looks correct, but every draft
token gets rejected: a 0% acceptance rate.

The patch is [wtdcode/vllm-backport pull request
#50](https://github.com/wtdcode/vllm-backport/pull/50), open and unmerged at
the time of writing. It reads the target model's embedding tensor from the checkpoint file. It
copies the tensor into the drafter's own embedding, so the drafter has real
values to work from. This repository adds one part the upstream patch does
not cover: `--tensor-parallel-size` above 1 shards the embedding table
across GPUs, and the upstream patch copies the full, unsharded tensor. It
fails with a size mismatch on `glm53flash6gpu`, which uses tensor parallel
and pipeline parallel together. The added code reads the shard boundaries
from vLLM's own embedding layer (`shard_indices`) and slices the checkpoint
tensor to match.

### glm53flash: tensor-parallel on 4 GPUs

`glm53flash` runs
[wtdcode/GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16)
with `--tensor-parallel-size 4`, matching wtdcode's own recipe, tested working
on this hardware. Start it with
[`run-glm53-flash-podman.sh`](scripts/run-glm53-flash-podman.sh). This
profile uses no pipeline parallel, so the target's embedding already sits on
every rank and the patch above does not apply to it.

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
second. When 500,000 tokens of context is enough and a 4-GPU footprint is
not a problem, use `glm53flash` over the profiles below.

### glm53flash6gpu: tensor and pipeline parallel on 6 GPUs, the fastest profile

`glm53flash6gpu` runs the same Flash checkpoint with `--tensor-parallel-size
2` and `--pipeline-parallel-size 3`. It splits the model into 3 pipeline
stages of 2 GPUs each. Start it with
[`run-glm53-flash-6gpu-podman.sh`](scripts/run-glm53-flash-6gpu-podman.sh).

This layout keeps every all-reduce inside one pair of GPUs. vLLM's custom
all-reduce kernel only works within a pair. It refuses to run across more
than 2 PCIe-only GPUs, so `glm53flash` (TP4) and `glm53flash8gpu` (PP8) both
fall back to NCCL's plain all-reduce instead.

The 3 pipeline stages split unevenly by default. The model spreads 45 hidden
layers across the stages, and the last stage also carries the MTP head and
the LM head. Those two add fixed memory on top of its share of the layers.
Setting `VLLM_PP_LAYER_PARTITION=16,15,14` puts fewer layers on the last
stage, which balances memory across all 3 stages. Without it, this profile
fails with an out-of-memory error before it starts sizing the KV cache, the
same problem `glm53flash` hit at `--pipeline-parallel-size 4`.

This profile mounts the patch described above and runs with MTP on. Draft
acceptance measures 54 to 78%, with a mean acceptance length of 2.6 to 3.3.
Single-stream decode measures about 54 tokens per second, the fastest of
every GLM-5.3 profile in this file, on 2 fewer GPUs than `glm53flash8gpu`
below. This profile only needs 6 of the 8 cards, leaving 2 free for other services.
When that matters more than the extra setup, use this profile over
`glm53flash`.

### glm53flash8gpu: pipeline-parallel on 8 GPUs

`glm53flash8gpu` runs the same model with `--pipeline-parallel-size 8`
instead, spreading the same weights across all 8 GPUs. Tested working on this
hardware: peak memory per card drops to about 52 GB out of 64 GB. That leaves enough room for `--max-model-len 1000000` (`GLM_MAXLEN`). That
value stays close to the model's native ceiling of `1048576`, and it matches
the limit set on every other model in this file. This profile also mounts
the MTP patch. Draft acceptance measures 54 to 78%, and single-stream decode
measures about 51 tokens per second. Start it with
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

### glm53int4 and glm53int48gpu: the full GLM-5.3 model, not Flash

`glm53int4` and `glm53int48gpu` run
[cyankiwi/GLM-5.3-AWQ-INT4](https://huggingface.co/cyankiwi/GLM-5.3-AWQ-INT4).
This is a different model from GLM-5.3-Flash above. It is zai-org's full
GLM-5.3, at 78 hidden layers against GLM-5.3-Flash's 45, published outside
the vllm-backport project. The checkpoint is 455 GB on disk.

`glm53int4` requests only 4 GPUs, 256 GB of VRAM total. The checkpoint alone
is 455 GB, so this profile cannot load the model at all, on any parallelism
setting.

`glm53int48gpu` requests all 8 GPUs, 512 GB total, and serves 262,144 tokens
of context. It uses `--tensor-parallel-size 2` on top of
`--pipeline-parallel-size 4`, so 4 stages of 2 GPUs each. It also uses
`--enable-expert-parallel`. Expert parallel splits this checkpoint's MoE
experts across the 2 GPUs in each stage. That halves the per-GPU expert
weight. Without it, each GPU in a stage holds close to the whole stage's
experts, and the model does not fit.

Start it with
[`run-glm53-int4-8gpu-podman.sh`](scripts/run-glm53-int4-8gpu-podman.sh).

#### Why the layout is 4 stages of 2 GPUs, and not 8 stages of 1

Pure pipeline parallel is the better layout on this hardware. The GPUs sit
on PCIe gen2 x4, so a tensor-parallel all-reduce on every layer costs more
than one activation tensor per stage boundary. It does not work with this
checkpoint.

This checkpoint's layers are not the same size. Layers 0 to 2 are dense and
take 0.75 GB each. Layers 4 to 76 take 5.4 GB each. **Layers 3 and 77 take
18.4 GB each**, because the quantization leaves them at higher precision.
Layer 78 is the grafted MTP block, at 18.5 GB.

Layer 77 sits on the last pipeline stage, and no partition can move it. On 8
stages that stage holds 59.3 GB of weight and then runs out of memory while
Marlin repacks layer 77, with 200 MB free. Tensor parallel plus expert
parallel is what makes it fit. Both split that layer's experts across the
2 GPUs of the stage, which halves the resident size and the repack
scratch.

#### Memory balance across the stages

vLLM allocates the same number of KV cache blocks on every rank. The context
limit is therefore the worst rank, at `free bytes / bytes per token`. Bytes
per token scale with the layer count on that rank. The target is therefore
free memory in proportion to layer count, not equal free memory.
`VLLM_PP_LAYER_PARTITION=20,20,20,18` gives the last stage 2 fewer layers,
because it also holds the 18.4 GB layer 77 and a separate, non-tied
`lm_head`.
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` avoids an allocator
fragmentation failure during Marlin weight repacking.

`--gpu-memory-utilization` does not limit weight loading itself. It only
sets a budget for the KV cache, sized after weight loading finishes.
Raising it from 0.85 towards 0.97 made no difference to whether the model
loads. It only changed how much KV cache was left over afterward.

#### The fp8 KV cache

This profile stores its KV cache in DeepSeek's packed `fp8_ds_mla` layout,
through `--kv-cache-dtype fp8_ds_mla` and `--attention-backend
TRITON_MLA_SPARSE`. The layout packs one token into 656 bytes: 512 e4m3
values, 4 group scales, and 64 unquantized RoPE values. The bf16 layout
needs 1152 bytes per token per layer, so the fp8 layout holds 1.76 times as
many tokens in the same memory.

Ampere has no fp8 hardware, and Triton refuses to name the `fp8e4nv` type
below SM89. The kernel therefore never names an fp8 type. It loads the cache
as `uint8` and rebuilds each value with integer shifts.

The KV pool holds 291,008 tokens at `--max-model-len 262144`. A
needle-in-haystack probe at 221,576 tokens returns the exact answer.

#### Speculative decoding is off

MTP speculative decoding costs about 6 GB on the last pipeline stage. That
stage is the one that binds the context limit, so the memory buys context
instead. Set `SPEC` in the launcher, or add `--speculative-config` back, to
trade context for decode speed. Batch-1 decode runs at about 21 tokens per
second without it. Prefill runs at about 1,340 tokens per second at 220,000
tokens of context.

#### Measured throughput

Every number below comes from `cyankiwi/GLM-5.3-AWQ-INT4` on all 8 GPUs, in
the `glm53int48gpu` configuration above. Speculative decoding is off. This
is the exact configuration:

```
VLLM_PP_LAYER_PARTITION=20,20,20,18
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GLM53_IDX_PREFILL_BUF_TOKENS=4000000

--tensor-parallel-size 2
--pipeline-parallel-size 4
--enable-expert-parallel
--disable-custom-all-reduce
--attention-backend TRITON_MLA_SPARSE
--kv-cache-dtype fp8_ds_mla
--block-size 64
--dtype bfloat16
--max-model-len 262144
--gpu-memory-utilization 0.97
--max-num-seqs 8
--enable-prefix-caching
```

Aggregate completion throughput, 256-token outputs, diverse short prompts:

| concurrent requests | aggregate tok/s | per stream |
|---|---|---|
| 1 | 21.5 | 21.5 |
| 2 | 41.3 | 20.7 |
| 4 | 67.0 | 16.8 |
| 8 | 77.0 | 9.6 |
| 16 | 69.2 | 4.3 |

Throughput peaks near 8 concurrent requests. Past that it drops, so
`--max-num-seqs 8` is the right ceiling. Four parallel sessions is the point
where each one still feels responsive.

Decode speed barely changes with context depth, which is what the sparse
top-k attention buys. With a warm prefix cache:

| workload | aggregate tok/s | per stream |
|---|---|---|
| 4 sessions at 64,444 tokens | 62.1 | 15.5 |
| 2 sessions at 139,955 tokens | 37.8 | 18.9 |

Prefill is the slow part. A cold 221,576-token prompt takes 165 seconds, at
1,343 tokens per second. Four cold 64,444-token prompts take 201 seconds
together, at 1,316 tokens per second. The prefix cache then makes a repeat
turn on the same context almost free: the same four sessions replay in 16
seconds.

Do not read these long-context numbers as one measurement. Prefills
serialize, so a single pass over cold prompts charges every later prefill to
the first stream's decode window. The table above measures decode on a
second, cache-warm pass.

#### The patch files

This profile needs five patch files, all under
[`patches/vllm-backport/`](patches/vllm-backport/), mounted read-only over
the vllm-backport image:

| patch | fixes |
|---|---|
| `deepseek_v32-fused-q-sm80-fp8.py` | this model's query-preprocessing kernel casts to a Triton fp8 type unsupported below SM89. Rewrites the cast to a software encoder that produces the same byte. Also keeps every cache buffer typed `uint8`, so Triton never has to declare the unsupported type |
| `mla-attention-sparse-mha-no-prefill.py` | three fixes. This model's attention backend has no dense-MHA prefill path, so it never sets a `prefill` field that a shared vLLM code path expects. The same backend never reaches the dense-MHA prefill code, so the patch also drops a 3.5 GB profile-run reserve for it. It also maps `--kv-cache-dtype fp8` onto `fp8_ds_mla` for this backend |
| `mla-fp8-sm80-kernel.py` | the sparse MLA attention kernel that reads the packed 656-byte fp8 cache without naming an fp8 Triton type. Vendored from [bayley/vllm-170hx-glm5](https://github.com/bayley/vllm-170hx-glm5) |
| `triton-mla-sparse-fp8.py` | declares `fp8_ds_mla` supported on the `TRITON_MLA_SPARSE` backend and routes decode to the kernel above |
| `indexer-prefill-buffer-cap.py` | the sparse indexer sizes its prefill gather workspace at 40 tokens per model token. That is a heuristic ceiling, not a requirement. The cap returns about 850 MB per GPU to the KV cache |

Ampere has no fp8 hardware, and Triton refuses to name the `fp8e4nv` type
below SM89. Triton also types a kernel parameter from the tensor dtype it
receives. A `float8_e4m3fn` view therefore fails compilation at the kernel
signature. Triton reports this as `at 1:0`. No runtime branch has to reach
the cast.
Keep every such buffer `torch.uint8`. On the host, once the kernel returns,
call `.view(torch.float8_e4m3fn)`. Both dtypes are 1 byte, so the view costs
nothing.
| `scheduler-pp-spec-stale-drafts.py` | drops draft tokens that did not ride on a request's own latest verified token. Only matters with speculative decoding on. See the MTP section below |

The fp8 kernel computes its cache offsets in int64. An int32 offset
overflows above about 3.27 million KV slots and faults with Xid 31, even
though every index value is in range. This box runs 1.4 million slots today,
so the cast is headroom rather than a live fix. Credit
[promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8),
which hit the same overflow on the bf16 kernel.

### glm53mix8gpu: the full GLM-5.3 at 1,048,576 tokens of context

`glm53mix8gpu` runs the same full GLM-5.3 as `glm53int48gpu`, from a
different quantization:
[Tech2wild/GLM-5.3-Int4-Int8Mix](https://huggingface.co/Tech2wild/GLM-5.3-Int4-Int8Mix).
The checkpoint is 377 GB on disk. It serves the model's **full 1,048,576-token
context** on all 8 GPUs, with a KV pool of 1,307,392 tokens, or 1.25 requests
at the full length.

#### Why this checkpoint and not the cyankiwi one

The two checkpoints hold the same model. They differ in which layers the
quantizer left alone, and that decides the whole layout.

| | cyankiwi/GLM-5.3-AWQ-INT4 | Tech2wild/GLM-5.3-Int4-Int8Mix |
|---|---|---|
| size on disk | 455 GB | 377 GB |
| a normal MoE layer | 5.4 GB | 4.8 GB |
| layer 3 | 18.4 GB, left in bf16 | 4.8 GB |
| layer 77 | 18.4 GB, left in bf16 | 4.8 GB |
| layer 78, the MTP block | 18.5 GB, left in bf16 | 9.4 GB, INT8 |

Layer 77 is the last layer, so it always lands on the last pipeline stage.
At 18.4 GB it leaves that stage no room to repack its own weights. The
cyankiwi checkpoint therefore cannot run 8 pipeline stages at all. The
Tech2wild checkpoint quantizes every layer from 1 to 77. All 8 stages then
carry 44 to 49 GB, and each one keeps 13 GB or more free for the KV cache.

Check this before you trust any new quantization of this model. Read
`quantization_config.ignore` in `config.json` and count the entries per
layer. A layer with about 780 ignored tensors keeps all 256 experts in
bf16 and takes 18.4 GB. A layer with 6 to 11 entries is quantized and takes
4.8 GB.

#### Layout

Pure pipeline parallel, 8 stages of 1 GPU, no tensor parallel and no expert
parallel. This is the right layout on PCIe gen2 x4. A pipeline hop ships one
activation tensor per stage boundary. Tensor parallel instead all-reduces on
every layer.

`VLLM_PP_LAYER_PARTITION=12,10,10,10,10,10,9,7` balances the stages by
memory, not by layer count. Stage 0 takes 12 layers because layers 0 to 2 are
dense and cost 0.4 to 0.8 GB each instead of 4.8 GB. Stage 7 takes 7 layers
because it also holds `lm_head` and the whole MTP block.

The KV cache uses the packed `fp8_ds_mla` layout, through the same patches
and the same `TRITON_MLA_SPARSE` backend as `glm53int48gpu` above.
Speculative decoding is off. See the section below for why.

#### Measured throughput

Aggregate completion throughput, 256-token outputs, diverse short prompts:

| concurrent requests | aggregate tok/s | per stream |
|---|---|---|
| 1 | 24.6 | 24.6 |
| 4 | 73.0 | 18.3 |
| 8 | 103.6 | 13.0 |

The CUDA graph capture sizes run up to 48 rather than the default 8. That
alone lifts the 8-stream number from 73.7 to 103.6, because a decode batch
wider than 8 otherwise falls back to eager.

Discard the first sweep after a restart. It reads 20 to 25% low while Triton
autotunes and the graphs warm up. The 8-stream figure settles at 101 to 103
over the next runs, and single-stream at 24.7.

Prefill runs at 1,330 to 2,370 tokens per second, faster on longer prompts.
A 204,819-token prompt takes 86 seconds and a cold 923,121-token prompt
takes 695 seconds. Treat the full million as a load-once batch mode, not an
interactive one. The prefix cache makes every later turn on the same context
cheap.

#### MTP speculative decoding does not work here yet

MTP is worth a lot on this model. At 3 draft tokens it takes single-stream
decode from 24.1 to about 43 tokens per second, and 8-stream from 73.7 to
90.7. Draft acceptance is healthy at 79%, 57%, and 39% across the three
draft positions, which is 2.76 tokens per model step.

It is off because it crashes on any prompt long enough to need several
prefill chunks. The failure is in vLLM, at
`vllm/v1/worker/gpu/model_runner.py`:

```
assert (num_scheduled_tokens_np >= num_logits).all()
```

The cause is the one
[bayley/vllm-170hx-glm5](https://github.com/bayley/vllm-170hx-glm5)
documents. Under pipeline parallel, several batches of one request are in
flight at once. Leftover `spec_token_ids` from an older batch make the
scheduler give the request more tokens than the step can hold. The crash
reproduces from 32,000 tokens upward. It is not caused by async scheduling,
by the prefill chunk budget, or by prefix caching.

`scheduler-pp-spec-stale-drafts.py` ports bayley's stale-draft guard. It
drops draft tokens that did not ride on the request's own latest verified
token. That guard alone does not fix the crash. bayley also serializes a
request against its own in-flight steps in the scheduler, and that part is
not ported. Enable MTP with `--speculative-config` only for short prompts.

Before you port more of that work, note one detail. bayley identifies a
settled request by `num_computed_tokens == num_tokens - 1`. This
vllm-backport image counts the just-sampled token as computed, so a settled
request here has `num_computed_tokens == num_tokens`. Ported verbatim, the
guard matches nothing, drops every draft, and turns MTP off while still
paying its cost. Single-stream then reads about 16 tokens per second,
below the 24.7 of plain decoding.

[promisezackr/glm53-flash-170hx-pp8](https://github.com/promisezackr/glm53-flash-170hx-pp8)
is the closer donor for that port. It runs the same `v1/worker/gpu/` layout
as this image. Its patch 0007 replaces boolean-mask draft indexing with a
Triton row-scatter, worth 2x single-stream. The mask path calls `nonzero()`,
which forces a device sync on every non-last rank every step. Patch 0021
adds adaptive per-request draft truncation. Patch 0022 removes it again.

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
