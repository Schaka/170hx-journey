# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM8x Triton stand-in for the fused DeepSeek V4.1 Q/KV prologue.

`torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` is compiled
into the wheel. The build on this host carries the DeepSeek V4.0 signature,
which takes 9 arguments and always applies the weightless Q RMSNorm. V4.1
passes a tenth argument, `apply_q_norm=False`, because it norms Q earlier with
a real weighted RMSNorm. Calling the 9-argument op would therefore norm Q
twice, and dropping the argument hides that rather than fixing it.

This module does the same work in Triton, so no CUDA source has to be rebuilt:

  Q  : optional per-head weightless RMSNorm, then GPT-J interleaved RoPE over
       the last ROPE_DIM channels, written into a head-padded output tensor
       whose padding heads are zero.
  KV : GPT-J interleaved RoPE, then UE8M0 FP8 quantization and paged insert
       through `quantize_and_insert_k_cache`, which already writes bytes and
       needs no fp8 Triton type.

Modelled on `vllm/models/deepseek_v4/xpu/xpu_qnorm_rope_kv_fp8_insert.py`,
which solves the same problem for XPU.
"""

import torch

from vllm.triton_utils import tl, triton

HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = HEAD_DIM - ROPE_DIM
HALF_ROPE = ROPE_DIM // 2


@triton.jit
def _qnorm_rope_kernel(
    q_ptr,  # [num_tokens, num_heads, HEAD_DIM] bf16, read
    q_out_ptr,  # [num_tokens, PADDED_HEADS, HEAD_DIM] bf16, written
    kv_ptr,  # [num_tokens, HEAD_DIM] bf16, read
    kv_out_ptr,  # [num_tokens, HEAD_DIM] bf16, written
    position_ids_ptr,
    cos_sin_cache_ptr,
    num_tokens,
    eps: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    PADDED_HEADS: tl.constexpr,
    APPLY_Q_NORM: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    """One program per (token, head), plus one trailing program for KV."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    if token_idx >= num_tokens:
        return

    offs = tl.arange(0, HEAD_DIM)
    rope_pair_idx = tl.arange(0, HALF_ROPE)
    even_offs = NOPE_DIM + rope_pair_idx * 2
    odd_offs = NOPE_DIM + rope_pair_idx * 2 + 1

    if head_idx >= PADDED_HEADS:
        # KV: GPT-J RoPE only. The caller quantizes and inserts.
        pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)
        cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(
            tl.float32
        )
        sin_val = tl.load(
            cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
        ).to(tl.float32)

        kv_base = kv_ptr + token_idx * HEAD_DIM
        kv_out_base = kv_out_ptr + token_idx * HEAD_DIM
        tl.store(kv_out_base + offs, tl.load(kv_base + offs))

        kv_even = tl.load(kv_base + even_offs).to(tl.float32)
        kv_odd = tl.load(kv_base + odd_offs).to(tl.float32)
        tl.store(
            kv_out_base + even_offs,
            (kv_even * cos_val - kv_odd * sin_val).to(kv_out_ptr.type.element_ty),
        )
        tl.store(
            kv_out_base + odd_offs,
            (kv_even * sin_val + kv_odd * cos_val).to(kv_out_ptr.type.element_ty),
        )
        return

    q_out_base = q_out_ptr + token_idx * PADDED_HEADS * HEAD_DIM + head_idx * HEAD_DIM

    if head_idx >= NUM_HEADS:
        # Padding head: FlashMLA reads the full padded width, so zero it.
        tl.store(
            q_out_base + offs,
            tl.zeros([HEAD_DIM], dtype=q_out_ptr.type.element_ty),
        )
        return

    pos = tl.load(position_ids_ptr + token_idx).to(tl.int64)
    cos_val = tl.load(cos_sin_cache_ptr + pos * ROPE_DIM + rope_pair_idx).to(tl.float32)
    sin_val = tl.load(
        cos_sin_cache_ptr + pos * ROPE_DIM + HALF_ROPE + rope_pair_idx
    ).to(tl.float32)

    q_base = q_ptr + token_idx * NUM_HEADS * HEAD_DIM + head_idx * HEAD_DIM
    q_vals = tl.load(q_base + offs).to(tl.float32)

    # One reciprocal for the whole head, derived from the raw values, then
    # applied to both the NoPE store and the RoPE pairs below.
    if APPLY_Q_NORM:
        rms = tl.rsqrt(tl.sum(q_vals * q_vals, axis=0) / HEAD_DIM + eps)
    else:
        rms = tl.full((), 1.0, dtype=tl.float32)

    # NoPE channels pass through; the RoPE channels are rewritten below.
    tl.store(
        q_out_base + offs,
        (q_vals * rms).to(q_out_ptr.type.element_ty),
        mask=offs < NOPE_DIM,
    )

    q_even = tl.load(q_base + even_offs).to(tl.float32) * rms
    q_odd = tl.load(q_base + odd_offs).to(tl.float32) * rms

    tl.store(
        q_out_base + even_offs,
        (q_even * cos_val - q_odd * sin_val).to(q_out_ptr.type.element_ty),
    )
    tl.store(
        q_out_base + odd_offs,
        (q_even * sin_val + q_odd * cos_val).to(q_out_ptr.type.element_ty),
    )


def qnorm_rope_kv_quant_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    swa_kv_cache_2d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    padded_heads: int,
    eps: float,
    block_size: int,
    apply_q_norm: bool,
) -> torch.Tensor:
    """Same contract as the fused op: returns padded, RoPE'd Q."""
    from vllm.models.deepseek_v4_1.common.ops.cache_utils import (
        quantize_and_insert_k_cache,
    )

    num_tokens, num_heads = q.shape[0], q.shape[1]
    q_out = torch.empty(
        (num_tokens, padded_heads, q.shape[2]), dtype=q.dtype, device=q.device
    )
    kv_roped = torch.empty_like(kv)
    if num_tokens:
        _qnorm_rope_kernel[(num_tokens, padded_heads + 1)](
            q,
            q_out,
            kv,
            kv_roped,
            positions,
            cos_sin_cache,
            num_tokens,
            eps=eps,
            NUM_HEADS=num_heads,
            PADDED_HEADS=padded_heads,
            APPLY_Q_NORM=apply_q_norm,
            HEAD_DIM=HEAD_DIM,
            ROPE_DIM=ROPE_DIM,
            NOPE_DIM=NOPE_DIM,
            HALF_ROPE=HALF_ROPE,
        )
        quantize_and_insert_k_cache(
            kv_roped, swa_kv_cache_2d, slot_mapping, block_size=block_size
        )
    return q_out
