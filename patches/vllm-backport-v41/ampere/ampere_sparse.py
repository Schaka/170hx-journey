# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100/A800/CMP 170HX).

Reuses the ROCm Triton sparse-MLA implementation wholesale, the same way
``vllm.models.deepseek_v4.ampere.ampere_sparse`` does for V4.0: its kernels,
ragged metadata builders, and bf16 o_proj reference path are plain
Triton/torch (the aiter-only preshuffle GEMMs self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.
"""

from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention,
    DeepseekV41ROCMAiterSparseSWABackend,
    DeepseekV4ROCMAiterMLASparseBackend,
)
from vllm.platforms.interface import DeviceCapability


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereSparseSWABackend(DeepseekV41ROCMAiterSparseSWABackend):
    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend
    swa_backend_cls = DeepseekV41AmpereSparseSWABackend
