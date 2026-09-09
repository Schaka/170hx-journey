# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-Triton sparse MLA backend for SM80 (A100) / SM121 (GB10)."""

import os
from typing import ClassVar

import torch

from vllm._glm53_mla_fp8 import triton_mla_sparse_attention_fp8
from vllm.utils.platform_utils import num_compute_units
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionLayer,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.xpu_mla_sparse import (
    XPUMLASparseBackend,
    XPUMLASparseImpl,
    XPUMLASparseMetadata,
    XPUMLASparseMetadataBuilder,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    flat_kv_row_view,
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.ops.triton_mla_sparse_kernel import (
    _DIM_QK,
    KV_SPLITS_CANDIDATES,
    triton_mla_sparse_attention,
)

# fp8_ds_mla packs one token into 656 bytes: 512 e4m3 NoPE latent values, 4
# fp32 group scales, then 64 bf16 RoPE values.
_FP8_DS_MLA_ENTRY_BYTES = 656


class TritonMLASparseMetadataBuilder(XPUMLASparseMetadataBuilder):
    # XPU base keeps NEVER (not validated under cudagraph); this subclass
    # claims UNIFORM_BATCH for the CUDA/Triton path.
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class TritonMLASparseImpl(XPUMLASparseImpl):
    """Triton sparse-MLA impl with split-KV decode (3-7× faster than the
    single-pass XPU base for single-query decode on SM80 / SM121)."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sm_count: int | None = None
        if self.topk_indices_buffer is not None:
            self._sm_count = num_compute_units(self.topk_indices_buffer.device.index)
        # GLM53_MLA_FP8=0 is an A/B-timing escape hatch only. It points the
        # bf16 kernel at a packed cache, which decodes to nonsense.
        self._fp8_kv = self.kv_cache_dtype == "fp8_ds_mla" and (
            os.environ.get("GLM53_MLA_FP8", "1") == "1"
        )
        self._warmup_autotune()

    def _warmup_autotune(self) -> None:
        """Prime `@triton.autotune` caches at init so the first request
        doesn't pay the inline config-sweep cost."""
        if self.topk_indices_buffer is None:
            return
        device = self.topk_indices_buffer.device
        topk = self.topk_indices_buffer.shape[-1]
        dim_qk = self.head_size
        q = torch.empty(1, self.num_heads, dim_qk, dtype=torch.bfloat16, device=device)
        indices = torch.zeros(1, 1, topk, dtype=torch.int32, device=device)
        if self._fp8_kv:
            kv_fp8 = torch.zeros(
                64, _FP8_DS_MLA_ENTRY_BYTES, dtype=torch.uint8, device=device
            )
            for splits in KV_SPLITS_CANDIDATES:
                triton_mla_sparse_attention_fp8(
                    q,
                    kv_fp8,
                    indices,
                    sm_scale=self.softmax_scale,
                    num_kv_splits=splits,
                    sm_count=self._sm_count,
                )
            return
        kv = torch.empty(64, 1, dim_qk, dtype=torch.bfloat16, device=device)
        for splits in KV_SPLITS_CANDIDATES:
            triton_mla_sparse_attention(
                q,
                kv,
                indices,
                sm_scale=self.softmax_scale,
                num_kv_splits=splits,
                sm_count=self._sm_count,
            )

    def _forward_bf16_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_c_and_k_pe_cache: torch.Tensor,  # [blocks, heads, d_qk]
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )
        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = triton_mla_sparse_attention(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output

    def _forward_fp8_kv(
        self,
        q: torch.Tensor,  # [sq, heads, d_qk]
        kv_rows: torch.Tensor,  # [rows, 656] uint8
        topk_indices: torch.Tensor,  # [sq, topk]
        attn_metadata: XPUMLASparseMetadata,
    ) -> torch.Tensor:
        num_tokens = q.shape[0]
        kv_rows = kv_rows.view(torch.uint8).reshape(-1, _FP8_DS_MLA_ENTRY_BYTES)
        output = triton_mla_sparse_attention_fp8(
            q,
            kv_rows,
            topk_indices.view(num_tokens, 1, -1),
            sm_scale=self.softmax_scale,
            sm_count=self._sm_count,
        )
        return output

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: XPUMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Copy of XPUMLASparseImpl.forward_mqa, minus its outright raise on a
        # quantized cache, plus the fp8 dispatch.
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        num_actual_toks = q.shape[0]

        buf = (
            self._indexer.topk_indices_buffer
            if self._indexer is not None
            else self.topk_indices_buffer
        )
        assert buf is not None, "topk_indices_buffer required for sparse MLA"
        topk_indices = buf[:num_actual_toks]

        kv_rows, block_stride_rows = flat_kv_row_view(
            kv_c_and_k_pe_cache, attn_metadata.block_size
        )
        topk_indices_global = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token,
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            BLOCK_STRIDE_ROWS=block_stride_rows,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        forward = self._forward_fp8_kv if self._fp8_kv else self._forward_bf16_kv
        attn_out = forward(q, kv_rows, topk_indices_global, attn_metadata)
        return attn_out, None


class TritonMLASparseBackend(XPUMLASparseBackend):
    """Same sparse-MLA contract as the XPU backend, CUDA Triton kernels."""

    supported_kv_cache_dtypes: ClassVar[list] = [
        "auto",
        "float16",
        "bfloat16",
        # DeepSeek's packed 656 B/token layout. Ampere has no fp8 hardware,
        # but none is needed: the compiled writer already emits this format on
        # SM80 and _glm53_mla_fp8.py decodes it from raw bytes.
        "fp8_ds_mla",
        "fp8",  # alias, canonicalized in mla_attention.py
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The DSA indexer backend requires block size 64 on CUDA and shares
        # the KV cache group with this backend; the base-class MultipleOf(1)
        # default lets auto-selection settle on 16, which then fails
        # select_common_block_size ("No common block size for 16").
        # MultipleOf(64) (rather than [64]) keeps larger user-specified
        # sizes like 128 usable, which measurably lowers profile-time peak
        # memory for very long contexts.
        return [MultipleOf(64)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # 576 = 512 latent + 64 RoPE (DeepSeek-V3.2 / GLM-5).
        # 512 = NoPE MLA (GLM-5.3-Flash, qk_rope_head_dim = 0).
        return [512, 576]

    @staticmethod
    def get_builder_cls() -> type["TritonMLASparseMetadataBuilder"]:
        return TritonMLASparseMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["TritonMLASparseImpl"]:
        return TritonMLASparseImpl
