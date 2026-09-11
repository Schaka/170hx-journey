# Model serving

Every backend starts through
[compose/docker-compose.yml](compose/docker-compose.yml). The compose file defines
fourteen profiles. The table below names every one of them. All fourteen are
mutually exclusive on this host, because they all bind port 8098.

## One served name, one context size

A client reads the context size from its own configuration. It cannot ask
which profile is running. So each `--served-model-name` names exactly one
context size. A profile that serves the same weights at a different size
carries the size in its name.

| served name | context | profile |
|---|---|---|
| `dsv4s` | 1,000,000 | `dsv4`, `dsv4backport` |
| `qwen3.8-flash-next` | 1,000,000 | `qwen`, `qwen8gpu` |
| `qwen3.8-flash-next-awq` | 1,000,000 | `qwenawq` |
| `glm-5.3` | 1,048,576 | `glm53mix8gpu` |
| `glm-5.3-flash` | 1,048,576 | `glm53flash6gpu` |
| `glm-5.3-flash-262k` | 262,144 | `glm53flash` |
| `glm-5.3-int4` | 262,144 | `glm53int48gpu` |
| `glm-5.3-int4-524k` | 524,288 | `glm53int4` |
| `deepseek-v4.1-flash` | 1,048,576 | `dsv41`, `dsv416`, `dsv416pp`, `dsv418` |

When the weights and the size both match, two profiles share a name.

## Model files

Model weights live on two RAID 0 arrays. See the storage section in
[HARDWARE.md](HARDWARE.md). The `/models` mount holds the models that this box
serves every day. The `/backup-models` mount holds the models that this box
keeps but rarely serves.

| model | path |
|---|---|
| Qwen3.8-Flash-Next-FP8 | `/models/Qwen` |
| Qwen3.8-Flash-Next-AWQ-W4A16 | `/models/Qwen3.8-Flash-Next-AWQ-W4A16` |
| GLM-5.3-Flash-AWQ-W4A16 | `/models/GLM-5.3-Flash-AWQ-W4A16` |
| GLM-5.3 (full model, INT4/INT8 mixed quant) | `/models/GLM-5.3-Int4-Int8Mix` |
| DeepSeek-V4.1-Flash | `/models/DeepSeek-V4.1-Flash` |
| DeepSeek-V4-Flash-0731 | `/backup-models/deepseek-ai/DeepSeek-V4-Flash-0731` |
| GLM-5.3 (full model, INT4 quant) | `/backup-models/GLM-5.3-AWQ-INT4` |

The compose file mounts these paths by default. Set `DSV4_MODEL`, `DSV41_MODEL`,
`QWEN_MODEL`, `QWEN_AWQ_MODEL`, `GLM_FLASH_MODEL`, `GLM_INT4_MODEL`, or
`GLM_MIX_MODEL` to override the path for a single run.

The `qwen`, `qwen8gpu`, and `qwenawq` services need a
`chat_template_lenient_system.jinja` file next to their model weights. That
file does not come from the model download. This repository keeps it at
[`templates/qwen3.8-flash-next/chat_template_lenient_system.jinja`](templates/qwen3.8-flash-next/chat_template_lenient_system.jinja).
Copy it into `/models/Qwen` and `/models/Qwen3.8-Flash-Next-AWQ-W4A16`.

The stock template refuses a system message that is not the first message:

```
System message must be at the beginning.
```

Agent clients send a system message in the middle of a conversation, so the
stock template rejects the request. The lenient template renders that message
as its own system block instead. Every other case produces the exact same
prompt as the stock template.

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

`glm53flash`, `glm53flash6gpu` and `glm53mix8gpu` below all use
pipeline parallel with MTP, and all mount a patch file over the
vllm-backport image:
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
not cover. `--tensor-parallel-size` above 1 shards the embedding table across
GPUs, and the upstream patch copies the full tensor. It
fails with a size mismatch on `glm53flash6gpu`, which uses tensor parallel
and pipeline parallel together. The added code reads the shard boundaries
from vLLM's own embedding layer (`shard_indices`) and slices the checkpoint
tensor to match.

### glm53flash: pipeline-parallel on 4 GPUs

`glm53flash` runs
[wtdcode/GLM-5.3-Flash-AWQ-W4A16](https://huggingface.co/wtdcode/GLM-5.3-Flash-AWQ-W4A16)
with `--pipeline-parallel-size 4`. Start it with
[`run-glm53-flash-podman.sh`](scripts/run-glm53-flash-podman.sh).

The 45 layers do not split evenly. The AWQ quantization leaves layer 45, the
MTP drafter, dense BF16. That layer lands on the last stage next to the LM
head. An even split overflows the last card by about 4 GiB before the engine
sizes the KV cache.

`VLLM_PP_LAYER_PARTITION=14,12,12,7` fixes that. It also balances the 11
`deepseek_sparse_attention` layers across the stages, as 3, 3, 3 and 2. Those
layers sit at indices 3, 7 and every fourth index up to 43. Do not rebalance
by layer count.
The layers differ in cost.

This profile mounts the MTP embedding patch described above, because the
drafter runs on the last stage under pipeline parallel.

Measured numbers at `--max-model-len 262144`:

| measure | value |
|---|---|
| KV cache pool | 1,670,327 tokens, 6.37x at full length |
| single-stream decode, warm | 71 tokens per second |
| 4 concurrent streams | 52 tokens per second total |
| 8 concurrent streams | 91 tokens per second total |

If 262,144 tokens of context is enough and 4 free GPUs matter, use
`glm53flash`. For a longer context, use `glm53flash6gpu` below. That profile
is both faster and reaches 1,048,576 tokens.

### glm53flash6gpu: tensor and pipeline parallel on 6 GPUs, the fastest profile

`glm53flash6gpu` runs the same Flash checkpoint with `--tensor-parallel-size
2` and `--pipeline-parallel-size 3`. It splits the model into 3 pipeline
stages of 2 GPUs each. Start it with
[`run-glm53-flash-6gpu-podman.sh`](scripts/run-glm53-flash-6gpu-podman.sh).

This layout keeps every all-reduce inside one pair of GPUs. vLLM's custom
all-reduce kernel only works within a pair. It refuses to run across more
than 2 PCIe-only GPUs, so a wider tensor-parallel group falls back to NCCL's
plain all-reduce instead.

The 3 pipeline stages split unevenly by default. The model spreads 45 hidden
layers across the stages, and the last stage also carries the MTP head and
the LM head. Those two add fixed memory on top of its share of the layers.
Setting `VLLM_PP_LAYER_PARTITION=16,15,14` puts fewer layers on the last
stage, which balances memory across all 3 stages. Without it, this profile
fails with an out-of-memory error before it starts sizing the KV cache.

This profile mounts the patch described above and runs with MTP on, at
`num_speculative_tokens` 2. Two is the value to use for prose. Code and
structured output can gain from 3. Five collapses draft acceptance and costs
speed on every workload.

This is the fastest profile in this file, and it reaches the model's full
context. Measured numbers at `--max-model-len 1048576`:

| measure | value |
|---|---|
| KV cache pool | 3,034,356 tokens, 2.89x at full length |
| single-stream decode, warm | 82 tokens per second |
| 4 concurrent streams | 54 tokens per second total |
| 8 concurrent streams | 91 tokens per second total |
| prefill at 209,786 tokens | 2,989 tokens per second |
| prefill at 917,169 tokens | 2,597 tokens per second |

A needle test at 917,169 tokens returns the exact answer, so the full context
works and does not only fit.

This layout beats `glm53flash` on speed because a token crosses 2 pipeline
hops instead of 3, and every all-reduce stays inside one GPU pair. If 6 cards
are free, use this profile.

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
context** on all 8 GPUs, with a KV pool of 1,411,264 tokens, or 1.35 requests
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

Tensor parallel loses on every axis here, even though it cuts the pipeline
from 7 hops to 3. Under tensor parallel each rank of a pair holds the whole
MLA KV cache for its stage, so the pool nearly halves. `glm53flash6gpu` gains
from tensor parallel because that model is smaller. This model has 78 layers
of hidden size 6144, so one all-reduce per layer costs more than 4 saved
hops.

| | PP8, TP1 | TP2, PP4 |
|---|---|---|
| KV pool | 1,411,264 to 1,426,240 tokens | 622,464 tokens |
| longest context | 1,048,576 | about 741,632 |
| prefill | 2,278 tok/s | 1,437 tok/s |
| decode, 1 stream | 25.2 to 25.4 tok/s | 24.5 tok/s |
| decode, 4 streams | 61.4 to 61.5 tok/s | 50.4 tok/s |
| decode, 8 streams | 83.3 to 86.1 tok/s | 61.4 tok/s |

`VLLM_PP_LAYER_PARTITION=12,10,10,10,10,10,9,7` balances the stages by
memory, not by layer count. Stage 0 takes 12 layers because layers 0 to 2 are
dense and cost 0.4 to 0.8 GB each instead of 4.8 GB. Stage 7 takes 7 layers
because it also holds `lm_head` and the whole MTP block.

The KV cache uses the packed `fp8_ds_mla` layout, through the same patches
and the same `TRITON_MLA_SPARSE` backend as `glm53int48gpu` above. MTP
speculative decoding runs at 3 draft tokens. Set `GLM_MIX_SPEC_K` to change
that count.

#### Two more patch files

This profile mounts the five patch files of `glm53int48gpu` above, plus
`mtp-embed-from-checkpoint.py` and these two:

| patch | fixes |
|---|---|
| `deepseek_v32-pp-topk-relay.py` | the model sets `index_topk_freq` to 4. Only layers 0, 1, 2 and every fourth layer after that run a full sparse indexer. Every other layer reuses the last selections from a buffer that belongs to one rank. A pipeline stage that starts on a reusing layer reads the previous batch's selections. The patch carries the selections over the pipeline hop and seeds the receiving rank's buffer. Set `GLM53_PP_TOPK_RELAY=0` to turn it off |
| `scheduler-pp-spec.py` | two scheduler fixes for MTP under pipeline parallel. It drops draft tokens that did not ride on a request's own latest verified token. It also holds a request back until its own sampling step lands. Set `GLM53_PP_SPEC_SERIALIZE=0` to turn the second one off |

The relay costs about 5 percent of aggregate decode at 8 streams, and about
1 percent single-stream. The price is the `[tokens, 2048]` int32 tensor that
each hop now carries. The KV pool does not shrink for it.

No layer split avoids the relay. To avoid it, every stage must start on a
full-indexer layer. Stage sizes must then be multiples of 4, which forces
stages of 12 dense-MoE layers. Those stages run out of memory on a 64 GB
card.

#### Measured throughput

Aggregate completion throughput, 512-token outputs, diverse short prompts,
staggered starts. `tok/step` counts the tokens that one model step returns,
so it shows how many MTP drafts the target accepts.

| concurrent requests | tok/s | per stream | tok/step | tok/s with MTP off |
|---|---|---|---|---|
| 1 | 25.4 | 25.4 | 2.43 | 19.7 |
| 4 | 61.5 | 15.4 | 2.30 | 54.3 |
| 8 | 86.1 | 10.8 | 2.39 | 72.7 |

MTP is worth 29 percent single-stream and 18 percent at 8 streams here. The
last column is the same server with `--speculative-config` removed.

Measure concurrency with long generations. A 256-token run at 8 streams
returns anywhere from 54 to 105 tokens per second on an unchanged server.
The first part of a generation runs faster than the steady state. A
512-token run still swings by up to 5 percent between runs, so read any
decode change below that as noise. Prefill is stable to 0.2 percent at a
fixed prompt length.

`GLM_MIX_SPEC_K` sets the MTP draft count, and 3 is the shipped value. At 2
the same server measures 25.2 to 26.0, 59.8 to 63.2, and 84.7 to 88.5 tokens
per second at 1, 4 and 8 streams. That is the same spread as 3, so this knob
is not worth turning on this model.

The CUDA graph capture sizes run up to 48 rather than the default 8. A
decode batch wider than 8 then keeps its graph instead of falling back to
eager.

The profile does not set `--max-num-batched-tokens`. Speculative decoding
makes vLLM pick 2048 and print this warning:

```
max_num_scheduled_tokens is set to 2048 based on the speculative decoding
settings. This may lead to suboptimal performance. Consider increasing
max_num_batched_tokens
```

Ignore that warning on this box. At 4096 the same server measures 2,240
tokens per second of prefill against 2,258. Decode drops to 22.3, 50.8 and
79.9 tokens per second at 1, 4 and 8 streams. The KV pool falls from
1,426,240 to 1,339,584 tokens. A wider chunk lengthens every pipeline bubble
across 8 stages on PCIe gen2 x4.

Prefill runs at 1,330 to 2,370 tokens per second, faster on longer prompts. A 204,819-token prompt
takes 86 seconds. A cold 844,617-token prompt takes 601 seconds. Treat the
full million as a load-once batch mode, not an interactive one. The prefix
cache makes every later turn on the same context cheap. The same
844,617-token context replays in 10 seconds.

#### What MTP needs under pipeline parallel

MTP crashes on this profile without the second fix in `scheduler-pp-spec.py`.
Any prompt long enough to need several prefill chunks trips an assertion in
`vllm/v1/worker/gpu/model_runner.py`:

```
assert (num_scheduled_tokens_np >= num_logits).all()
```

Under pipeline parallel, several batches of one request are in flight at
once. Leftover `spec_token_ids` from an older batch make the scheduler give
the request more tokens than the step can hold. The crash starts at about
32,000 tokens of prompt.

The fix holds a request back until its own sampling step lands, which is the
serialization that plain decoding gets for free. It gates on sampling steps
only. A pure prefill chunk ships no sampled token and no draft, so those
chunks still pipeline across the stages. bayley reports that serializing
them too costs 8 times on prefill.

The guard identifies a settled request by `num_computed_tokens ==
num_tokens`. bayley's original uses `num_tokens - 1`, because his vLLM does
not count the just-sampled token as computed. Ported without that change,
the guard matches nothing and drops every draft.

With both fixes the profile runs 24 copy-fidelity requests from 8,139 to
124,652 tokens of context with no error and no engine death.

## DeepSeek-V4.1-Flash

The `dsv41` profile serves DeepSeek-V4.1-Flash on all 8 GPUs at the model's
full 1,048,576-token context. The model has a 552-billion-parameter backbone
in 40 layers. It activates 8 billion parameters in prefill and 16 billion in
decode. It uses CSA2, a compressed sparse attention that stores 890 bytes of
key-value data per token. It also carries Engram, a conditional memory of 196
billion parameters that the runtime keeps in host memory.

### The image

No public image runs this model on sm_80. The build kit in
[patches/vllm-backport-v41/](patches/vllm-backport-v41/) makes one.
[`build.sh`](patches/vllm-backport-v41/build.sh) does five steps:

1. Copy the `vllm` tree out of the `lazymio/vllm-backport:v0.12.0-sm80` image.
2. Apply the vLLM pull request 56201 diff with `git apply --reject`.
3. Run [`fixups.py`](patches/vllm-backport-v41/fixups.py), which repairs every
   hunk the backport tree rejects, and adds the sm_80 changes below.
4. Copy the three files in [`ampere/`](patches/vllm-backport-v41/ampere/) into
   the tree.
5. Build the image as `localhost/vllm-backport-v41:sm80`.

The kit needs a vLLM checkout with the pull request fetched:

```bash
git clone --filter=blob:none https://github.com/vllm-project/vllm.git
cd vllm && git fetch https://github.com/vllm-project/vllm.git pull/56201/head:pr56201
cd ~/v41kit && VLLM_SRC=~/vllm bash build.sh
```

Every edit in `fixups.py` finds its place by a text anchor and runs twice with
the same result. A missing anchor stops the run with the file name, so an
upstream change fails loudly instead of building a wrong image.

### Layout: 4-way tensor parallel across 2 pipeline stages

The model configuration sets `kv_source_layer_ids` to `[2, 8, 14, 20]`. Layers that
share one compressed key-value cache form a group. The groups are layers 0 to
1, 2 to 7, 8 to 13, 14 to 19, and 20 to 39. vLLM refuses to split a group
across pipeline stages:

```
NotImplementedError: PP splits inside a v4.1 kv-sharing group are not supported
```

The last group holds 20 of the 40 layers. Those 20 layers must sit on one
stage. That caps the pipeline at 2 stages, so 8 GPUs means 4-way tensor
parallel inside each stage. `VLLM_PP_LAYER_PARTITION` is `20,20`.

This layout suits the PCIe topology. Each PEX8749 switch carries 4 GPUs, and
vLLM numbers ranks tensor-parallel first. Tensor-parallel group 0 lands on
GPUs 0 to 3 and group 1 on GPUs 4 to 7. Every all-reduce stays inside one
switch. Only the pipeline hand-off crosses between switches, and that is one
hidden-state tensor per micro-batch.

### The sm_80 gaps the kit closes

- Triton types a kernel parameter from the tensor dtype and rejects `fp8e4nv`
  below SM89. The kit keeps those buffers as `torch.uint8` and converts in
  software through `vllm/v1/attention/ops/fp8_sm80.py`.
- `dequantize_and_gather_k_cache` picks its path with `has_cutedsl()`, which
  only asks whether the package is installed. The CuteDSL kernels need SM90,
  so sm_80 takes that path and the compiler aborts. The kit swaps in
  `is_cutedsl_supported()`, which also tests the compute capability.
- The compiled `_C` operator
  `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` carries the V4.0
  signature and takes 9 arguments. V4.1 passes a tenth, `apply_q_norm`.
  Dropping it normalizes Q twice and gives wrong output with no error.
  [`qnorm_rope_kv_insert.py`](patches/vllm-backport-v41/ampere/qnorm_rope_kv_insert.py)
  replaces the operator with a Triton kernel that takes the flag.
- Marlin packs 4 fp8 values into one int32, so the `wo_a` weight arrives with
  the wrong shape. The kit routes that one layer to the emulation kernel.
- The key-value cache grouping code asserts a `[MLA, *SWA]` layer order and
  drops the 3 compressor circular buffers. The kit gives those buffers their
  own group and appends it after the arithmetic.
- Under pipeline parallel the DSpark drafter runs on the last stage and owns
  no embedding table. It aliases the target table, which the target only
  builds on the first stage. When a DSpark drafter is
  configured, the kit builds that table on the last stage as well. The weight loader keys off the
  parameter existing, so the weights arrive by themselves.

### Speculative decoding with DSpark

DSpark is the model's own speculator. The checkpoint carries 3 next-token
layers that target layers 37 to 39. `DSV41_SPEC_K` sets the draft count and 5
is the shipped value. At 3 the same server measures 41.4, 73.8 and 103.3
tokens per second at 1, 4 and 8 streams. That is the same spread as 5, so
this knob is not worth turning on this model.

The extra embedding table on the last stage costs 272,501 tokens of key-value
pool, which is under 2 percent.

### Measured throughput

Aggregate completion throughput, 512-token outputs, diverse short prompts,
staggered starts. `tok/step` counts the tokens that one model step returns,
so it shows how many DSpark drafts the target accepts.

| concurrent requests | tok/s | per stream | tok/step | tok/s with DSpark off |
|---|---|---|---|---|
| 1 | 43.8 | 43.8 | 2.85 | 26.9 |
| 4 | 72.9 | 18.2 | 2.31 | 63.6 |
| 8 | 104.2 | 13.0 | 2.42 | 97.5 |

DSpark is worth 63 percent single-stream and 7 percent at 8 streams here. The
last column is the same server with `--speculative-config` removed.

The key-value pool holds 14,058,003 tokens. That is 13.41 concurrent requests
at the full 1,048,576-token context.

Prefill runs at 1,584, 1,650 and 1,389 tokens per second for prompts of
39,028, 119,028 and 319,028 tokens.

### Acceptance depends on the prompt

DSpark acceptance moves more with prompt content than with any setting. One
stream, 512-token outputs, greedy sampling:

| prompt class | accepted of drafted | tok/s |
|---|---|---|
| repetitive text | 0.594 | 79.4 |
| code | 0.606 | 78.5 |
| descriptive prose | 0.316 | 52.2 |
| technical prose | 0.291 | 49.6 |

Read any single decode number against the prompt that produced it. The
synthetic filler in the concurrency benchmark is high-entropy text, which is
the worst case for a speculator.

### Reasoning output for an agent client

Both V4.1 profiles set three parser flags:

```
--reasoning-parser deepseek_v41
--enable-auto-tool-choice
--tool-call-parser deepseek_v41
```

Without the reasoning parser, the thinking block stays inside the `content`
field. An agent client then writes that text back into the next turn. Over a
long session the model reads its own thinking as conversation and drifts into
a repetition loop.

The parser alone is not enough. vLLM names the field `reasoning`. The DeepSeek
API and most other providers name it `reasoning_content`, and that is the name
agent clients look for, opencode included. The build kit adds the second name
as an alias at both places that serialize a message:

- `vllm/entrypoints/openai/chat_completion/protocol.py`, for the complete
  message of a request that does not stream.
- `vllm/entrypoints/openai/engine/protocol.py`, for every streamed delta. A
  client that reads each chunk needs the name there too.

Both names carry the same text, so a client that reads either one works. The
six-card profile answers with both:

```
fields: ['annotations', 'audio', 'content', 'function_call', 'reasoning',
         'reasoning_content', 'refusal', 'role']
```

```
"delta":{"reasoning":"We","reasoning_content":"We"}
```

### dsv416: the same model on 6 GPUs

The `dsv416` profile serves DeepSeek-V4.1-Flash on GPUs 0 to 5, with
tensor-parallel-size 2 and pipeline-parallel-size 3. It frees 2 cards for
other work and it prefills faster than the 8-card profile. It holds far less
key-value cache, so it suits single long sessions rather than many parallel
ones.

| measure | 6 GPUs | 8 GPUs |
|---|---|---|
| decode, 1 stream | 43.3 tok/s | 43.8 tok/s |
| decode, 4 streams | 85.5 tok/s | 72.9 tok/s |
| prefill, 119,025 tokens | 2,551 tok/s | 1,650 tok/s |
| key-value pool | 3,049,911 tokens | 14,058,003 tokens |
| concurrency at 1,048,576 tokens | 2.91x | 13.41x |

Prefill gains because tensor-parallel-size 2 splits each all-reduce over 2
cards inside one PLX switch, in place of 4. Decode at 4 streams gains for the
same reason. The key-value pool falls because the same weights sit on 6 cards
in place of 8, which leaves less free memory on each one.

#### Splitting a kv-sharing group across pipeline stages

Six cards needs a pipeline cut inside the layer 20 to 39 group. That group
weighs about 154.6 GiB and does not fit on 2 cards. Layer 20 publishes three
things that every later layer in the group reads:

- the compressed key-value cache,
- the indexer key cache,
- the candidate blocks that every later indexer masks its scores against.

[`pp_kv_group_relay.py`](patches/vllm-backport-v41/ampere/pp_kv_group_relay.py)
gives the third stage its own copy of both caches and refills them each step.
The payload is the compressor latent, `[num_tokens, 512]` in bfloat16. Both
cache writes are pure functions of that latent, the token positions, the
rotary cache and the slot mapping, so the third stage recomputes them. The
candidate blocks ride the same pipeline hop as a second tensor.

The third stage also needs the `wk` and `k_norm` weights of the layer 20
indexer, about 64 thousand parameters, to derive the index keys. The relay
registers them under the names the checkpoint uses, which are relative to the
inner model and not the forward-context path.

Layer 20 is a `PPMissingLayer` on the third stage, so
`is_pp_missing_parameter` reports every name under it as absent and the
loader skips it. The kit consults the relay alias names before it trusts that
report.

#### Choosing the layer partition

`VLLM_PP_LAYER_PARTITION` is `14,14,12`. The cut at layer 14 falls on a
kv-sharing group boundary. The cut at layer 28 falls inside the layer 20 to
39 group, which the relay supports. The last stage takes 2 layers fewer
for a reason. It also carries the head, the DSpark drafter and the embedding
table that DSpark needs there. Those add about 5.1 GiB.

Weights land at 50.32, 49.45 and 47.77 GiB per card. A partition of
`14,12,14` puts 54.70 GiB on the last stage and the engine then reports:

```
ValueError: No available memory for the cache blocks.
```

The profile sets the memory fraction to 0.95, not 0.92. Six cards leave a
much thinner margin than 8.

#### Recall test

Both profiles recall a fact planted at 15 percent and at 75 percent depth in
prompts of 25,073 and 103,073 tokens. Send `thinking` and `reasoning_effort`
in `chat_template_kwargs` on every request. V4.1 reads a numeric budget from
1 to 100, where `low` is 25, `high` is 50, `xhigh` is 75 and `max` is 100.
Without a value the model uses 50 and can drift into a repetition loop on
long context.

### dsv416pp: one pipeline stage per card on 6 GPUs

The `dsv416pp` profile serves DeepSeek-V4.1-Flash on GPUs 0 to 5 with
tensor-parallel-size 1 and pipeline-parallel-size 6. It is the fastest V4.1
profile on this box. It prefills twice as fast as `dsv416` and three times as
fast as `dsv41`, and it decodes faster at every stream count.

| measure | dsv416pp | dsv416 | dsv41 |
|---|---|---|---|
| GPUs | 6 | 6 | 8 |
| layout | TP1 x PP6 | TP2 x PP3 | TP4 x PP2 |
| decode, 1 stream | 46.6 tok/s | 43.3 tok/s | 43.8 tok/s |
| decode, 4 streams | 106.6 tok/s | 85.5 tok/s | 72.9 tok/s |
| decode, 8 streams | 153.1 tok/s | not measured | 104.2 tok/s |
| prefill, 119,027 tokens | 5,078 tok/s | 2,551 tok/s | 1,650 tok/s |
| key-value pool | 2,196,629 tokens | 3,049,911 tokens | 14,058,003 tokens |
| concurrency at 1,048,576 tokens | 2.09x | 2.91x | 13.41x |

Prefill gains because the profile runs no all-reduce at all. Every all-reduce
on this box crosses a Gen2 x4 link at about 1.6 GB/s. At tensor-parallel-size
4 each layer moves about 16 MB per 2,048-token chunk, which sets a ceiling
near 1,700 tokens per second. The pipeline hop carries the hidden states and
the relay payload alone, which is far less traffic.

The profile also prefills 39,027 tokens at 4,928 tokens per second and
319,027 tokens at 4,209 tokens per second.

#### The layer partition

`VLLM_PP_LAYER_PARTITION` is `7,7,7,7,7,5`. Weights land at 51.27, 49.61,
49.79, 49.60, 49.60 and about 45.7 GiB. The last stage takes 5 layers,
because it also holds the head, the DSpark drafter and the extra embedding
table. Those weigh about 10.2 GiB together, which is more than one layer.
A partition of `7,7,7,7,6,6` puts about 52.9 GiB on the last stage and the
engine then reports:

```
ValueError: To serve at least one request with the model's max seq len
```

#### What a stage needs from the stage before it

Five of the six stages start on a layer that writes neither of the two
caches it reads. The pipeline hop therefore carries four payloads beside the
hidden states:

- the compressor latent, which refills the replicated compressed cache,
- the candidate blocks, which every later indexer masks its scores against,
- the top-k indices, which a layer with no indexer of its own reads,
- the `pre_mix` tensor the model already sent.

The top-k indices matter because the model runs an indexer at layers 2, 8,
14, 20, 24, 28, 32 and 36 only. Every other layer reads the indices the last
one published. A cut between an indexer layer and its readers leaves those
readers with an empty buffer. They then read the wrong part of the context.

A stage that writes no kv source of its own passes the latent it receives
on to the next stage. The candidate blocks chain the same way, through the
buffer each stage refills.

#### Recall test

The profile recalls a fact planted at 15 percent and at 75 percent depth in
prompts of 25,073, 103,073, 259,073 and 649,073 tokens. All eight runs pass.

#### Long context costs prefill, not decode

Decode on a cached prompt of 259,079 tokens runs at 46.7 tokens per second.
That matches the rate at 2,000 tokens. The indexer scores the whole context
at layers 24, 28, 32 and 36 on every step. That work does not show in the
decode time on this hardware.

Prefill is where length costs. The profile prefills 119,027 tokens at 5,078
tokens per second and 779,069 tokens at 2,917 tokens per second.

### dsv418: one pipeline stage per card

The `dsv418` profile serves DeepSeek-V4.1-Flash on all 8 GPUs with
tensor-parallel-size 1 and pipeline-parallel-size 8. It carries no measured
numbers yet.

The profile exists to test one claim. Every all-reduce on this box crosses a
Gen2 x4 link. At tensor-parallel-size 4 each layer moves about 16 MB per
2,048-token chunk over a 1.6 GB/s link. That is about 15 ms per all-reduce
and about 1.2 s per chunk. It works out near 1,700 tokens per second. The
`dsv41` profile measures that prefill. Prefill on that profile is
therefore bound by the link and not by the cards. A pipeline stage per card
removes the all-reduce and sends the hidden states alone.

The cost is the key-value pool. Tensor parallel splits the compressed cache
across the cards of a stage, and a pure pipeline does not. The pool will land
near the 6-GPU figure rather than the 8-GPU one.

`VLLM_PP_LAYER_PARTITION` is `5,5,5,5,5,5,5,5`. Only the cut at layer 20
falls on a kv-sharing group boundary. The other six cuts fall inside a group,
so six stages run a relay:

| stage | layers | reads a group written on | relay |
|---|---|---|---|
| 0 | 0 to 4 | itself | none |
| 1 | 5 to 9 | layer 2 | rebuilds from the hop |
| 2 | 10 to 14 | layer 8 | rebuilds from the hop |
| 3 | 15 to 19 | layer 14 | rebuilds from the hop |
| 4 | 20 to 24 | itself | none |
| 5 | 25 to 29 | layer 20 | rebuilds and passes on |
| 6 | 30 to 34 | layer 20 | rebuilds and passes on |
| 7 | 35 to 39 | layer 20 | rebuilds from the hop |

Stages 5 and 6 write no kv source of their own. The hop carries one latent
slot, filled by the last kv source on the sending stage. A stage with no kv
source copies the latent it receives into that slot and sends it on. The
candidate blocks chain the same way, through the buffer each stage refills.

Only the cut at layer 20 falls on an indexer layer, so six stages also read
the top-k indices from the hop. See the `dsv416pp` section above for what
each payload does.

The last stage carries the head, the DSpark drafter and the extra embedding
table, about 5.1 GiB on top of its 5 layers. If it reports `ValueError: No
available memory for the cache blocks`, set
`DSV418_PARTITION=6,5,5,5,5,5,5,4` to move a layer off it.

### Reasoning reaches the client under both field names

Every V4.1 profile sets `--reasoning-parser deepseek_v41`, `--tool-call-parser
deepseek_v41` and `--enable-auto-tool-choice`. Without the reasoning parser
the thinking block stays inside `content`. An agent client then writes that
text back into the next turn, and the model drifts into a repetition loop.

The build kit adds one more edit for the same problem. vLLM names the
reasoning field `reasoning`. The DeepSeek API and most other providers name
it `reasoning_content`, and that is the name agent clients look for, opencode
included. The kit emits both names, in the complete message and in every
streamed delta. A client that reads either one works.

This is the same fix as patch `0023` in the
[fork](https://github.com/Schaka/deepseek-v4-cmp170hx), written for this
codebase.

### A minimum number of tokens holds the thinking block open

The V4.1 generation prompt ends with the `<think>` token, so every turn starts
inside the thinking block. The checkpoint can close that block with its first
generated token. The reasoning block is then empty. The model writes its
deliberation into `content` instead. The real `</think>` at the end of that
deliberation reaches the parser in its content state, where the parser absorbs
it without an event. The client shows the thinking as chat text, and it writes
that text back into the next turn.

The parser cannot repair this. It already streamed the text as content, so it
cannot relabel it. The fix belongs at sampling time.

`patches/vllm-backport-v41/min_thinking_tokens.py` masks the `</think>` token
until the request reaches `VLLM_MIN_THINKING_TOKENS` output tokens. The
`dsv416pp` profile mounts it and sets 32. It tracks only a request whose
prompt ends inside a thinking block, so a request that asks for no thinking is
untouched.

The module subclasses `MinTokensLogitsProcessor`. The rejection sampler
applies only that class under speculative decoding. A plain `LogitsProcessor`
runs in the normal sampler and stops running the moment DSpark is on.

### The parser recovers a tool call that opens inside the thinking block

When a request omits the `thinking` flag, the model means thinking. The
parser therefore starts each turn in its reasoning state. The kit adds two
recovery paths for that state.

The first path handles a lost envelope. The model can write a well-formed
`<｜DSML｜ invoke ...>` block without the `<｜DSML｜ calls>` wrapper that
opens it. The parser then starts the tool call from the invoke marker alone,
and it accepts only a tool name the request declares. The base tree does this
from the content state. The kit adds the same path from the reasoning state.
It closes the reasoning block first, so the thinking text stays reasoning and
the call does not land inside it.

The second path handles a near-miss envelope. Near the context ceiling the
model can drop the space in the wrapper and write `<｜DSML｜calls>`. The kit
accepts that spelling as the wrapper. The lexer matches the longest literal
first, so the correct wrapper behaves the same as before.

The third path handles a lost special token. `｜DSML｜` is one token in the
vocabulary. The model can drop it and write the markers in plain text, and it
can mix the two spellings inside one call. This is a real example:

```
<tool_calls><invoke name="Bash"><parameter name="command">pwd</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>
```

The kit adds `<invoke name="`, `</invoke>` and `</tool_calls>` as markers. The
parser starts the call from the plain invoke marker. It closes the call on
either spelling. It accepts only a tool name the request declares. The
parameter patterns now read either spelling on each side of a value, and the
`string="true"` attribute is optional. A value that carries no attribute
parses as JSON first, then as plain text.

The kit does not add a plain `<tool_calls>` wrapper marker. A bare wrapper
carries no tool name to check. Text that talks about the protocol can then
consume the rest of the message.

Without the third path the whole call reaches the client as text, and the
thinking block never closes.

The first two paths are the same fixes as patches `0016` and `0018` in the
fork, written for this codebase.

### The profile does not set `--max-num-batched-tokens`

Speculative decoding makes vLLM pick 2048 and print this warning:

```
max_num_scheduled_tokens is set to 2048 based on the speculative decoding
settings. This may lead to suboptimal performance. Consider increasing
max_num_batched_tokens
```

Ignore that warning on this box. At 8192 the same server measures 1,574 and
1,673 tokens per second of prefill against 1,584 and 1,650, which is inside
the run-to-run spread. Decode drops to 39.1, 68.9 and 93.6 tokens per second
at 1, 4 and 8 streams. The key-value pool falls from 14,058,003 to 8,121,690
tokens. Prefill on this box is bound by the work inside one chunk, not by the
number of chunks.

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
because all twelve profiles share port 8098.

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
  (GLM-5.3-Flash-AWQ-W4A16, 4 GPUs).
- `run-glm53-flash-6gpu-podman.sh` starts the `glm53flash6gpu` profile
  (GLM-5.3-Flash-AWQ-W4A16, 6 GPUs).
- `run-glm53-int4-podman.sh` starts the `glm53int4` profile
  (GLM-5.3-AWQ-INT4, 4 GPUs). Does not start on this hardware today.
- `run-glm53-int4-8gpu-podman.sh` starts the `glm53int48gpu` profile
  (GLM-5.3-AWQ-INT4, all 8 GPUs).
- `run-glm53-mix-8gpu-podman.sh` starts the `glm53mix8gpu` profile
  (GLM-5.3-Int4-Int8Mix, all 8 GPUs).
- `run-dsv41-podman.sh` starts the `dsv41` profile
  (DeepSeek-V4.1-Flash, all 8 GPUs).
- `run-dsv41-6gpu-podman.sh` starts the `dsv416` profile
  (DeepSeek-V4.1-Flash, 6 GPUs).
- `run-dsv41-6gpu-pp-podman.sh` starts the `dsv416pp` profile
  (DeepSeek-V4.1-Flash, 6 GPUs, one pipeline stage per card).
- `run-dsv41-pp8-podman.sh` starts the `dsv418` profile
  (DeepSeek-V4.1-Flash, all 8 GPUs, one pipeline stage per card).
