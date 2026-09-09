# SPDX-License-Identifier: Apache-2.0
"""Sparse MLA attention over an fp8_ds_mla KV cache, for sm_80.

Vendored from bayley/vllm-170hx-glm5 (glm52_mla_fp8.py, Apache-2.0) and
mounted into vllm-backport as `vllm/_glm53_mla_fp8.py`.

Ampere has no fp8 hardware and Triton refuses to even *name* `fp8e4nv` below
sm_89 ("type fp8e4nv not supported in this architecture") -- the same wall the
DSA indexer hit (see patch note 5 in the serving memory). But fp8 here is only
a *storage* format: nothing needs an fp8 tensor core, only a decode from bytes
to something the bf16 MMA can eat. So this kernel never mentions an fp8 type.
It loads the cache as `uint8` and reconstructs values with integer ops.

The decode trick
----------------
An e4m3 byte and an fp16 word have the same field *order* (sign, exponent,
mantissa), so a single shift lands every field where fp16 wants it::

    u16 = ((b & 0x7F) << 7) | ((b & 0x80) << 8)

Reading that back as fp16 gives exactly the e4m3 value divided by 256, for
normals *and* subnormals alike:

  * normal (e>0):  fp16 = 2^(e-15) * (1 + m/8),  e4m3 = 2^(e-7) * (1 + m/8)
  * subnormal:     fp16 = m * 2^-17,             e4m3 = m * 2^-9

so one exponent-bias correction of 2^8 covers both cases, and it is *free*
because it folds into the dequantization scale that has to be applied anyway.
No branches, no select, no subnormal special case. (e4m3 NaN, 0x7F/0xFF,
decodes to +-496 rather than NaN. A KV cache holding NaN is already lost, and
not propagating it is the friendlier failure.)

The layout (verified byte-for-byte against `ops.concat_and_cache_mla`)
---------------------------------------------------------------------
656 bytes per token, which is what vLLM's `fp8_ds_mla` cache dtype already
allocates and what the *compiled* `concat_and_cache_mla` already writes --
that op turns out to work fine on sm_80, so the write side needs no changes::

    [  0: 512]  512 x e4m3   NoPE latent (kv_c), scaled per 128-element group
    [512: 528]    4 x fp32   group scales, = group_amax / 448
    [528: 656]   64 x bf16   RoPE (k_pe), stored unquantized

Because 656 is divisible by 4 and by 2, the same allocation can be addressed
as uint8 (stride 656), fp32 (stride 164, scales at 128..131) and bf16
(stride 328, RoPE at 264..327). Passing all three views costs nothing and
keeps the index arithmetic honest.

Where the scales go
-------------------
The scale varies per (token, 128-dim group), which would normally mean
materializing a [512, BLOCK_N] scale tile just to multiply it into the loaded
values. Instead both dots are split into their four 128-dim groups so the
scale is applied to a *vector*, never a tile:

  * QK: `qk += sc[g] * dot(q_g, k_g_raw)` -- the group is the contraction
    axis, so the scale factors out of the dot entirely and lands on the
    [BLOCK_H, BLOCK_N] result as a row broadcast. Applied in fp32.
  * PV: the group is the *output* axis, so it cannot factor out the same way;
    instead `p` is pre-scaled per group (`p * sc[g]`, a [BLOCK_N] broadcast)
    and each group accumulates into its own [BLOCK_H, 128] tile.

This is not just cheaper than a scale tile, it is more accurate than
dequantizing into bf16 would be: the raw e4m3 values carry 4 significant bits
and convert to bf16 *exactly*, so the only rounding left is the one the bf16
baseline already pays when it rounds `p` for the PV dot.
"""

import functools

import torch

from vllm.triton_utils import LOG2E, LOGE2, tl, triton
from vllm.utils.platform_utils import num_compute_units

# fp8_ds_mla shape constants. DeepSeek-V3.2 / GLM-5 fix all of these.
_BLOCK_DMODEL = 512  # NoPE latent (kv_lora_rank)
_BLOCK_DPE = 64  # RoPE (qk_rope_head_dim)
_BLOCK_DV = 512  # V is the NoPE latent
_DIM_QK = _BLOCK_DMODEL + _BLOCK_DPE  # 576

_QUANT_GROUP = 128
_NUM_GROUPS = _BLOCK_DMODEL // _QUANT_GROUP  # 4
_ENTRY_BYTES = 656

# Element offsets of each region, per view.
_U8_STRIDE = _ENTRY_BYTES  # 656
_F32_STRIDE = _ENTRY_BYTES // 4  # 164
_BF16_STRIDE = _ENTRY_BYTES // 2  # 328
_F32_SCALE_OFF = _BLOCK_DMODEL // 4  # 128
_BF16_ROPE_OFF = (_BLOCK_DMODEL + 16) // 2  # 264

# Folds the 2^8 exponent-bias correction of the fp16 decode into the scale.
_E4M3_DECODE_BIAS = 256.0

# Triton refuses plain module globals inside @jit ("can only access global
# variables instantiated as constexpr"), while host code needs plain ints for
# shapes and asserts. Mirror the layout constants above rather than threading
# ten more constexpr arguments through every call site -- they are fixed by the
# fp8_ds_mla format, not tuned.
K_DMODEL = tl.constexpr(_BLOCK_DMODEL)
K_DPE = tl.constexpr(_BLOCK_DPE)
K_DV = tl.constexpr(_BLOCK_DV)
K_GROUP = tl.constexpr(_QUANT_GROUP)
K_U8_STRIDE = tl.constexpr(_U8_STRIDE)
K_F32_STRIDE = tl.constexpr(_F32_STRIDE)
K_BF16_STRIDE = tl.constexpr(_BF16_STRIDE)
K_F32_SCALE_OFF = tl.constexpr(_F32_SCALE_OFF)
K_BF16_ROPE_OFF = tl.constexpr(_BF16_ROPE_OFF)
K_DECODE_BIAS = tl.constexpr(_E4M3_DECODE_BIAS)

_BLOCK_H = 16
_MIN_BLOCK_N = 16

_MERGE_BLOCK_H = 1
_MERGE_BLOCK_DV_TILE = 128
_NUM_MERGE_DV_TILES = _BLOCK_DV // _MERGE_BLOCK_DV_TILE

# Wider BLOCK_N sweep than the bf16 kernel's, which is pinned to 16 (final) and
# 32 (split). Decoding e4m3 adds per-tile work that the bf16 path does not pay
# -- eight group scale loads, the shift/or/bitcast chain -- so bigger tiles have
# more to amortize and the bf16 kernel's tuning does not transfer.
_FINAL_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bn in (16, 32, 64)
    for nw in (2, 4, 8)
    for ns in (2, 4)
]
_SPLIT_AUTOTUNE_CONFIGS = [
    triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bn in (32, 64)
    for nw in (4, 8)
    for ns in (2, 4)
]

KV_SPLITS_CANDIDATES = (1, 2, 4, 8, 16)
_MIN_TOPK_PER_SPLIT = 128
_SPLIT_MAX_OCCUPANCY = 4


@triton.jit
def _e4m3_bytes_to_bf16(b):
    """Decode e4m3 bytes to bf16 holding `value / 256`. See module docstring.

    The 1/256 is not corrected here -- callers fold it into the group scale.
    """
    u = b.to(tl.uint16)
    packed = ((u & 0x7F) << 7) | ((u & 0x80) << 8)
    # bf16 keeps every e4m3 value exactly: 4 significant bits into 8.
    return packed.to(tl.float16, bitcast=True).to(tl.bfloat16)


@triton.jit
def _load_q_group(q_buffer, cur_q, cur_head, mask_h, g, stride_q_token, stride_q_head):
    offs = g * K_GROUP + tl.arange(0, K_GROUP)
    return tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + offs[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )


@triton.jit
def _load_kv_group(kv_u8, indices, mask_kv, g, stride_kv_token, TRANSPOSED: tl.constexpr):
    """Load one 128-dim group of the NoPE latent for BLOCK_N tokens.

    TRANSPOSED=True  -> [BLOCK_N, 128], token-major, for the PV dot.
    TRANSPOSED=False -> [128, BLOCK_N], dim-major, for the QK dot.
    """
    offs = g * K_GROUP + tl.arange(0, K_GROUP)
    if TRANSPOSED:
        ptr = kv_u8 + indices[:, None] * stride_kv_token + offs[None, :]
        raw = tl.load(ptr, mask=mask_kv[:, None], other=0)
    else:
        ptr = kv_u8 + indices[None, :] * stride_kv_token + offs[:, None]
        raw = tl.load(ptr, mask=mask_kv[None, :], other=0)
    return _e4m3_bytes_to_bf16(raw)


@triton.jit
def _sparse_mla_compute_tile_fp8(
    q_buffer,
    kv_u8,  # uint8 view  [num_slots, 656]
    kv_f32,  # fp32 view   [num_slots, 164]
    kv_bf16,  # bf16 view   [num_slots, 328]
    indices_ptr,
    cur_q,
    cur_head,
    cur_kv_head_id,
    mask_h,
    split_start,
    split_end,
    seq_kv,
    stride_q_token,
    stride_q_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """fp8_ds_mla twin of `_sparse_mla_compute_tile`.

    Same online-softmax structure and the same NEG_LARGE sentinel; the only
    differences are the dequantizing loads and the four per-group PV
    accumulators (see module docstring).
    """
    offs_dpe = tl.arange(0, K_DPE)
    offs_g = tl.arange(0, K_GROUP)

    # Q is loop-invariant: load all four NoPE groups plus RoPE once.
    q0 = _load_q_group(q_buffer, cur_q, cur_head, mask_h, 0, stride_q_token, stride_q_head)
    q1 = _load_q_group(q_buffer, cur_q, cur_head, mask_h, 1, stride_q_token, stride_q_head)
    q2 = _load_q_group(q_buffer, cur_q, cur_head, mask_h, 2, stride_q_token, stride_q_head)
    q3 = _load_q_group(q_buffer, cur_q, cur_head, mask_h, 3, stride_q_token, stride_q_head)
    qpe = tl.load(
        q_buffer
        + cur_q * stride_q_token
        + cur_head[:, None] * stride_q_head
        + (K_DMODEL + offs_dpe)[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )

    # Finite sentinel (not -inf) -- when an entire BLOCK_N tile is masked,
    # `-inf - -inf = NaN` poisons the softmax; `sentinel - sentinel = 0`
    # gives `exp2(0) = 1` and the matching V rows are already 0.
    NEG_LARGE = -1.0e30
    e_max = tl.zeros([BLOCK_H], dtype=tl.float32) + NEG_LARGE
    e_sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    # One accumulator per quantization group: the PV scale lives on the output
    # axis, so the groups cannot share a single [BLOCK_H, 512] tile.
    acc0 = tl.zeros([BLOCK_H, K_GROUP], dtype=tl.float32)
    acc1 = tl.zeros([BLOCK_H, K_GROUP], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_H, K_GROUP], dtype=tl.float32)
    acc3 = tl.zeros([BLOCK_H, K_GROUP], dtype=tl.float32)

    for start_indice in range(split_start, split_end, BLOCK_N):
        offs_indice = start_indice + tl.arange(0, BLOCK_N)
        mask_indice = offs_indice < split_end
        indices = tl.load(
            indices_ptr
            + cur_q * stride_indices_token
            + cur_kv_head_id * stride_indices_head
            + offs_indice,
            mask=mask_indice,
            other=-1,
        )
        mask_kv = (indices >= 0) & (indices < seq_kv)
        safe_idx = tl.where(mask_kv, indices, 0)

        # Group scales: [BLOCK_N] each, with the fp16-decode bias folded in.
        sc_base = kv_f32 + safe_idx * K_F32_STRIDE + K_F32_SCALE_OFF
        sc0 = tl.load(sc_base + 0, mask=mask_kv, other=0.0) * K_DECODE_BIAS
        sc1 = tl.load(sc_base + 1, mask=mask_kv, other=0.0) * K_DECODE_BIAS
        sc2 = tl.load(sc_base + 2, mask=mask_kv, other=0.0) * K_DECODE_BIAS
        sc3 = tl.load(sc_base + 3, mask=mask_kv, other=0.0) * K_DECODE_BIAS

        # -- QK. The group is the contraction axis, so each group's scale is a
        # plain row broadcast on the fp32 dot result.
        k0 = _load_kv_group(kv_u8, safe_idx, mask_kv, 0, K_U8_STRIDE, False)
        k1 = _load_kv_group(kv_u8, safe_idx, mask_kv, 1, K_U8_STRIDE, False)
        k2 = _load_kv_group(kv_u8, safe_idx, mask_kv, 2, K_U8_STRIDE, False)
        k3 = _load_kv_group(kv_u8, safe_idx, mask_kv, 3, K_U8_STRIDE, False)
        qk = tl.dot(q0, k0) * sc0[None, :]
        qk += tl.dot(q1, k1) * sc1[None, :]
        qk += tl.dot(q2, k2) * sc2[None, :]
        qk += tl.dot(q3, k3) * sc3[None, :]

        # RoPE is stored unquantized, so this is an ordinary bf16 load.
        kpe = tl.load(
            kv_bf16
            + safe_idx[None, :] * K_BF16_STRIDE
            + (K_BF16_ROPE_OFF + offs_dpe)[:, None],
            mask=mask_kv[None, :],
            other=0.0,
        )
        qk += tl.dot(qpe, kpe)

        qk *= sm_scale
        qk = tl.where((mask_h[:, None]) & (mask_kv[None, :]), qk, NEG_LARGE)

        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp2(e_max - n_e_max)
        p = tl.exp2(qk - n_e_max[:, None])

        # -- PV. The group is the output axis: pre-scale `p` per group instead.
        v0 = _load_kv_group(kv_u8, safe_idx, mask_kv, 0, K_U8_STRIDE, True)
        v1 = _load_kv_group(kv_u8, safe_idx, mask_kv, 1, K_U8_STRIDE, True)
        v2 = _load_kv_group(kv_u8, safe_idx, mask_kv, 2, K_U8_STRIDE, True)
        v3 = _load_kv_group(kv_u8, safe_idx, mask_kv, 3, K_U8_STRIDE, True)
        acc0 = acc0 * re_scale[:, None] + tl.dot((p * sc0[None, :]).to(tl.bfloat16), v0)
        acc1 = acc1 * re_scale[:, None] + tl.dot((p * sc1[None, :]).to(tl.bfloat16), v1)
        acc2 = acc2 * re_scale[:, None] + tl.dot((p * sc2[None, :]).to(tl.bfloat16), v2)
        acc3 = acc3 * re_scale[:, None] + tl.dot((p * sc3[None, :]).to(tl.bfloat16), v3)

        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    return acc0, acc1, acc2, acc3, e_max, e_sum


@triton.autotune(configs=_FINAL_AUTOTUNE_CONFIGS, key=["index_topk", "kv_group_num"])
@triton.jit
def _sparse_mla_fp8_kernel_final(
    q_buffer,
    kv_u8,
    kv_f32,
    kv_bf16,
    indices_ptr,
    out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_out_token,
    stride_out_head,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Single-pass fast path: full topk, write final bf16 output directly."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    acc0, acc1, acc2, acc3, e_max, e_sum = _sparse_mla_compute_tile_fp8(
        q_buffer,
        kv_u8,
        kv_f32,
        kv_bf16,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        0,
        index_topk,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
    )

    # Guard against queries with zero valid KV (e_sum == 0 -> NaN from 0/0).
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_g = tl.arange(0, K_GROUP)
    base = out_ptr + cur_q * stride_out_token + cur_head[:, None] * stride_out_head
    tl.store(base + (0 * K_GROUP + offs_g)[None, :],
             (acc0 / e_sum_safe[:, None]).to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(base + (1 * K_GROUP + offs_g)[None, :],
             (acc1 / e_sum_safe[:, None]).to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(base + (2 * K_GROUP + offs_g)[None, :],
             (acc2 / e_sum_safe[:, None]).to(tl.bfloat16), mask=mask_h[:, None])
    tl.store(base + (3 * K_GROUP + offs_g)[None, :],
             (acc3 / e_sum_safe[:, None]).to(tl.bfloat16), mask=mask_h[:, None])


@triton.autotune(
    configs=_SPLIT_AUTOTUNE_CONFIGS,
    key=["index_topk", "NUM_KV_SPLITS", "kv_group_num"],
)
@triton.jit
def _sparse_mla_fp8_kernel_split(
    q_buffer,
    kv_u8,
    kv_f32,
    kv_bf16,
    indices_ptr,
    mid_out_ptr,
    seq_kv,
    h_q,
    stride_q_token,
    stride_q_head,
    stride_mid_token,
    stride_mid_head,
    stride_mid_split,
    stride_indices_token,
    stride_indices_head,
    sm_scale,
    index_topk: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    kv_group_num: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    LOGE2: tl.constexpr,
):
    """Stage 1 of split-KV: one slice of the topk axis -> (out, lse) partials."""
    cur_q = tl.program_id(0)
    cur_head_id = tl.program_id(1)
    split_kv_id = tl.program_id(2)
    cur_kv_head_id = cur_head_id // tl.cdiv(kv_group_num, BLOCK_H)

    VALID_BLOCK_H: tl.constexpr = BLOCK_H if kv_group_num > BLOCK_H else kv_group_num
    cur_head = cur_head_id * VALID_BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = (cur_head < (cur_head_id + 1) * VALID_BLOCK_H) & (cur_head < h_q)

    split_topk: tl.constexpr = tl.cdiv(index_topk, NUM_KV_SPLITS)
    split_start = split_kv_id * split_topk
    split_end = tl.minimum(split_start + split_topk, index_topk)

    acc0, acc1, acc2, acc3, e_max, e_sum = _sparse_mla_compute_tile_fp8(
        q_buffer,
        kv_u8,
        kv_f32,
        kv_bf16,
        indices_ptr,
        cur_q,
        cur_head,
        cur_kv_head_id,
        mask_h,
        split_start,
        split_end,
        seq_kv,
        stride_q_token,
        stride_q_head,
        stride_indices_token,
        stride_indices_head,
        sm_scale,
        BLOCK_H,
        BLOCK_N,
    )

    # When a split has no valid KV (`e_sum == 0`), guard the divide so the mid
    # buffer holds 0 instead of NaN; otherwise the `0 * NaN = NaN` term in
    # stage 2 would poison every other split.
    e_sum_safe = tl.where(e_sum > 0, e_sum, 1.0)
    offs_g = tl.arange(0, K_GROUP)
    mid_base_2d = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head[:, None] * stride_mid_head
        + split_kv_id * stride_mid_split
    )
    tl.store(mid_base_2d + (0 * K_GROUP + offs_g)[None, :],
             acc0 / e_sum_safe[:, None], mask=mask_h[:, None])
    tl.store(mid_base_2d + (1 * K_GROUP + offs_g)[None, :],
             acc1 / e_sum_safe[:, None], mask=mask_h[:, None])
    tl.store(mid_base_2d + (2 * K_GROUP + offs_g)[None, :],
             acc2 / e_sum_safe[:, None], mask=mask_h[:, None])
    tl.store(mid_base_2d + (3 * K_GROUP + offs_g)[None, :],
             acc3 / e_sum_safe[:, None], mask=mask_h[:, None])
    mid_lse_ptr = (
        mid_out_ptr
        + cur_q * stride_mid_token
        + cur_head * stride_mid_head
        + split_kv_id * stride_mid_split
        + K_DV
    )
    tl.store(mid_lse_ptr, (e_max + tl.log2(e_sum)) * LOGE2, mask=mask_h)


@functools.lru_cache(maxsize=256)
def _choose_num_kv_splits(
    num_tokens: int, num_head_groups: int, index_topk: int, sm_count: int
) -> int:
    """Same heuristic as the bf16 kernel; see triton_mla_sparse_kernel.py."""
    import os as _os

    _force = int(_os.environ.get("GLM52_MLA_SPLITS", "0") or 0)
    if _force:
        return _force
    baseline = num_tokens * num_head_groups
    if baseline == 0 or baseline >= sm_count * 2:
        return 1
    ideal = triton.next_power_of_2(max(1, index_topk // _MIN_TOPK_PER_SPLIT))
    max_splits = max(1, (sm_count * 2) // baseline)
    max_splits = 1 << (max_splits.bit_length() - 1)
    num_kv_splits = min(ideal, max_splits)
    while num_kv_splits > 1 and index_topk % num_kv_splits != 0:
        num_kv_splits //= 2
    return max(1, num_kv_splits)


def triton_mla_sparse_attention_fp8(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    num_kv_splits: int | None = None,
    sm_count: int | None = None,
) -> torch.Tensor:
    """Sparse MLA attention over an fp8_ds_mla (uint8, 656B/token) KV cache.

    Args:
        q:         [num_tokens, num_heads_q, 576] bf16
        kv:        [num_slots, 656] uint8, the fp8_ds_mla packed cache
        indices:   [num_tokens, 1, topk] int32, global slot indices
        sm_scale:  softmax scale
        num_kv_splits: override auto-heuristic; None/0 = auto, 1 = single-pass.
        sm_count:  cached device SM count for the split heuristic.

    Returns:
        out: [num_tokens, num_heads_q, 512] bf16
    """
    num_tokens, num_heads_q, dim_qk = q.shape
    assert dim_qk == _DIM_QK, (
        f"sparse MLA kernel requires dim_qk={_DIM_QK} (DeepSeek-V3.2 / GLM-5), "
        f"got {dim_qk}"
    )
    kv = kv.view(-1, _ENTRY_BYTES)
    assert kv.dtype == torch.uint8 and kv.is_contiguous(), (
        f"fp8_ds_mla cache must be a contiguous uint8 tensor, got {kv.dtype}"
    )
    index_topk = indices.shape[-1]
    assert index_topk % _MIN_BLOCK_N == 0, (
        f"topk ({index_topk}) must be a multiple of the smallest autotune "
        f"BLOCK_N ({_MIN_BLOCK_N})"
    )

    # Three views of one allocation. 656 is divisible by 4 and 2, and torch
    # allocations are 256B-aligned, so both reinterpretations are legal.
    kv_u8 = kv
    kv_f32 = kv.view(torch.float32)
    kv_bf16 = kv.view(torch.bfloat16)
    seq_kv = kv.shape[0]

    kv_group_num = num_heads_q
    num_head_groups = triton.cdiv(num_heads_q, min(_BLOCK_H, kv_group_num))

    if num_kv_splits is None or num_kv_splits == 0:
        if sm_count is None:
            sm_count = num_compute_units(q.device.index)
        num_kv_splits = _choose_num_kv_splits(
            num_tokens, num_head_groups, index_topk, sm_count
        )

    out = torch.empty(
        (num_tokens, num_heads_q, _BLOCK_DV), dtype=torch.bfloat16, device=q.device
    )

    if num_kv_splits == 1:
        _sparse_mla_fp8_kernel_final[(num_tokens, num_head_groups)](
            q_buffer=q,
            kv_u8=kv_u8,
            kv_f32=kv_f32,
            kv_bf16=kv_bf16,
            indices_ptr=indices,
            out_ptr=out,
            seq_kv=seq_kv,
            h_q=num_heads_q,
            stride_q_token=q.stride(0),
            stride_q_head=q.stride(1),
            stride_out_token=out.stride(0),
            stride_out_head=out.stride(1),
            stride_indices_token=indices.stride(0),
            stride_indices_head=indices.stride(1),
            sm_scale=sm_scale * LOG2E,
            index_topk=index_topk,
            kv_group_num=kv_group_num,
            BLOCK_H=_BLOCK_H,
        )
        return out

    mid_out = torch.empty(
        (num_tokens, num_heads_q, num_kv_splits, _BLOCK_DV + 1),
        dtype=torch.float32,
        device=q.device,
    )
    _sparse_mla_fp8_kernel_split[(num_tokens, num_head_groups, num_kv_splits)](
        q_buffer=q,
        kv_u8=kv_u8,
        kv_f32=kv_f32,
        kv_bf16=kv_bf16,
        indices_ptr=indices,
        mid_out_ptr=mid_out,
        seq_kv=seq_kv,
        h_q=num_heads_q,
        stride_q_token=q.stride(0),
        stride_q_head=q.stride(1),
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_indices_token=indices.stride(0),
        stride_indices_head=indices.stride(1),
        sm_scale=sm_scale * LOG2E,
        index_topk=index_topk,
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_BLOCK_H,
        LOGE2=LOGE2,
    )

    # Stage 2 is dtype-agnostic (fp32 partials), so reuse the bf16 kernel's.
    from vllm.v1.attention.ops.triton_mla_sparse_kernel import _sparse_mla_merge_kernel

    _sparse_mla_merge_kernel[(num_tokens, num_heads_q, _NUM_MERGE_DV_TILES)](
        mid_out_ptr=mid_out,
        out_ptr=out,
        h_q=num_heads_q,
        stride_mid_token=mid_out.stride(0),
        stride_mid_head=mid_out.stride(1),
        stride_mid_split=mid_out.stride(2),
        stride_out_token=out.stride(0),
        stride_out_head=out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        kv_group_num=kv_group_num,
        BLOCK_H=_MERGE_BLOCK_H,
        BLOCK_DV=_BLOCK_DV,
        BLOCK_DV_TILE=_MERGE_BLOCK_DV_TILE,
        num_warps=2,
    )
    return out
