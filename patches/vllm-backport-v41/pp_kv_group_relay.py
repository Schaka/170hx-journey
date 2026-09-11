"""Relay for a DeepSeek-V4.1 kv-sharing group that a pipeline cut divides.

Layers 20 to 39 all read one compressed KV cache and one indexer K cache that
layer 20 writes. Stock vLLM refuses a pipeline cut inside that range, because
the later stage then holds readers but no writer. That refusal forces the
whole 20-layer group onto one pipeline stage, which in turn forces 8 GPUs.

This module lifts the restriction. The later stage gets its own copy of both
caches and refills them each step from one tensor sent over the pipeline hop.

The payload is the compressor latent, `[num_tokens, head_dim]` in bfloat16.
Both cache writes are pure functions of that latent, the token positions, the
rotary cache and the slot mapping. The later stage already has the last three,
so it recomputes both writes locally. It needs the source indexer `wk` and
`k_norm` weights to derive the index keys. Those are small, about 64 thousand
parameters, and the weight loader fills them by prefix like any other weight.

At 1,024 bytes per token the payload is under 5 percent of the hidden states
that the same pipeline hop already carries, so the relay is close to free.
"""

from typing import Any, cast

import torch
from torch import nn

from vllm.config import CacheConfig, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.models.deepseek_v4_1.common.ops import indexer_k_norm_rope_store
from vllm.models.deepseek_v4_1.common.ops.fused_compress_quant_cache import (
    rope_quant_insert,
)
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    get_kv_quant_mode,
)

logger = init_logger(__name__)


class CompressedKVReplica(nn.Module, AttentionLayerBase):
    """Stands in for the kv-source attention layer on a stage that lacks it.

    Consumers read their source through `static_forward_context[prefix].kv_cache`.
    This module answers that lookup and reports the same cache specification
    the real source layer reports, so the cache manager allocates a tensor of
    the same shape and the metadata builder produces a slot mapping for it.
    """

    def __init__(
        self,
        prefix: str,
        head_dim: int,
        block_size: int,
        cache_dtype_str: str,
        torch_dtype: torch.dtype,
        compress_ratio: int,
        backend_cls: type[AttentionBackend],
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.prefix = prefix
        self.head_dim = head_dim
        self.block_size = block_size
        self.cache_dtype_str = cache_dtype_str
        self.torch_dtype = torch_dtype
        self.compress_ratio = compress_ratio
        self.backend_cls = backend_cls

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C], the same squeeze the real layer does.
        self.kv_cache = kv_cache.squeeze(1)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        uses_fp8_ds_mla_layout = self.cache_dtype_str == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=torch.uint8 if uses_fp8_ds_mla_layout else self.torch_dtype,
            tokens_per_state=self.compress_ratio,
            cache_dtype_str=self.cache_dtype_str,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_dtype_str),
            state_content_bytes=584 if uses_fp8_ds_mla_layout else None,
        )

    def forward(self):  # pragma: no cover - never called
        raise NotImplementedError


class SourceIndexerShadow(nn.Module):
    """The `wk` and `k_norm` of the kv-source indexer, on a stage that lacks it.

    The index key is `k_norm(wk(latent))`, RoPE'd at the group boundary. Both
    weights are named with the source layer prefix, so the checkpoint loader
    fills them without any mapping change.
    """

    def __init__(
        self,
        prefix: str,
        head_dim: int,
        index_head_dim: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.wk = ReplicatedLinear(
            head_dim,
            index_head_dim,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.wk",
        )
        self.k_norm = RMSNorm(index_head_dim, eps=rms_norm_eps)


class KVGroupRelay(nn.Module):
    """Owns the replicated caches and refills them from the relayed latent.

    One instance per pipeline stage that reads a group it does not write.
    """

    def __init__(
        self,
        source_attn_prefix: str,
        replica: CompressedKVReplica,
        index_k_cache: Any,
        shadow: SourceIndexerShadow,
        compress_ratio: int,
        use_fp4_kv: bool,
        rotary_emb: nn.Module,
    ):
        super().__init__()
        self.source_attn_prefix = source_attn_prefix
        self.rotary_emb = rotary_emb
        self.replica = replica
        self.index_k_cache = index_k_cache
        self.shadow = shadow
        self.compress_ratio = compress_ratio
        self.use_fp4_kv = use_fp4_kv

    def write(
        self,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
    ) -> None:
        """Repeat both of the source's cache writes on this stage.

        Call this once per step, before any consumer layer of the group runs.
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or latent is None:
            # Profile run: the caches are not bound yet.
            return


        # 1. The compressed main cache. Same call the source compressor makes.
        main_metadata = cast(Any, attn_metadata[self.source_attn_prefix])
        rope_quant_insert(
            latent,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.replica.kv_cache,
            main_metadata.slot_mapping,
            self.compress_ratio,
            fp8_scale=None,
        )

        if self.index_k_cache is None:
            # This stage starts on a layer that runs no indexer of its own.
            # It reads the top-k indices the earlier stage published, which
            # travel over the hop, so there is no indexer key cache to fill.
            return

        # 2. The indexer K cache. Rows at non-boundary tokens hold garbage
        # latent and the store kernel skips them, the same as on the source.
        index_metadata = cast(Any, attn_metadata[self.index_k_cache.prefix])
        k_pre, _ = self.shadow.wk(latent)
        indexer_k_norm_rope_store(
            k_pre,
            positions,
            self.rotary_emb.cos_sin_cache,
            self.shadow.k_norm.weight,
            self.shadow.k_norm.variance_epsilon,
            self.index_k_cache.kv_cache,
            index_metadata.slot_mapping,
            self.compress_ratio,
            self.use_fp4_kv,
        )


def build_relay(
    consumer_attn: Any,
    source_attn_prefix: str,
    index_k_cache: Any,
    cache_config: CacheConfig,
    config: Any,
) -> KVGroupRelay:
    """Build the replica set from the first consumer layer on this stage.

    Every layer in one sharing group has the same head dimension, compression
    ratio and cache dtype, so the consumer describes the absent source.
    """
    replica = CompressedKVReplica(
        prefix=source_attn_prefix,
        head_dim=consumer_attn.head_dim,
        block_size=cache_config.block_size,
        cache_dtype_str=consumer_attn.kv_cache_dtype,
        torch_dtype=consumer_attn.kv_cache_torch_dtype,
        compress_ratio=consumer_attn.compress_ratio,
        backend_cls=consumer_attn.backend_cls,
    )
    consumer_attn._static_forward_context[source_attn_prefix] = replica
    shadow = SourceIndexerShadow(
        prefix=f"{source_attn_prefix}.indexer",
        head_dim=consumer_attn.head_dim,
        index_head_dim=config.index_head_dim,
        rms_norm_eps=config.rms_norm_eps,
    )
    relay = KVGroupRelay(
        source_attn_prefix=source_attn_prefix,
        replica=replica,
        index_k_cache=index_k_cache,
        shadow=shadow,
        compress_ratio=consumer_attn.compress_ratio,
        use_fp4_kv=getattr(consumer_attn, "use_fp4_kv", False),
        rotary_emb=consumer_attn.rotary_emb,
    )
    _PENDING.append(relay)
    return relay


# Layer construction runs before the model can collect the relays it built, so
# `build_relay` parks them here and the model takes them right after it builds
# its layers. Construction is sequential, so a plain list is enough.
_PENDING: list[KVGroupRelay] = []


def take_relays() -> list[KVGroupRelay]:
    """Return the relays built since the last call, and clear the list."""
    relays = list(_PENDING)
    _PENDING.clear()
    return relays


def _checkpoint_base(source_attn_prefix: str) -> str:
    """Turn a forward-context prefix into the name the checkpoint uses.

    The forward context keys layers by their full path, such as
    `language_model.model.layers.20.attn`. The weight loader works with names
    relative to the inner model, such as `layers.20.attn`, because it builds
    its parameter dictionary from that model's own `named_parameters`.
    """
    cut = source_attn_prefix.find("layers.")
    base = source_attn_prefix[cut:] if cut >= 0 else source_attn_prefix
    return f"{base}.indexer"


def alias_names(relays: list[KVGroupRelay]) -> set[str]:
    """The checkpoint names the shadow weights answer to."""
    names: set[str] = set()
    for relay in relays:
        base = _checkpoint_base(relay.source_attn_prefix)
        names.add(f"{base}.wk.weight")
        names.add(f"{base}.k_norm.weight")
    return names


def alias_source_params(relays: list[KVGroupRelay], params_dict: dict) -> None:
    """Let the shadow weights load under their source-layer checkpoint names.

    The shadow lives under the model as `kv_group_relays.N.shadow.*`, but the
    checkpoint names it `layers.<source>.attn.indexer.wk.weight`. Add that
    name to the parameter dictionary so the stock loader finds it.

    The name alone is not enough. The source layer is a `PPMissingLayer` on
    this stage, so `is_pp_missing_parameter` reports every name under it as
    absent and the loader skips it. The caller must consult `alias_names`
    before it trusts that report.
    """
    for relay in relays:
        base = _checkpoint_base(relay.source_attn_prefix)
        params_dict[f"{base}.wk.weight"] = relay.shadow.wk.weight
        params_dict[f"{base}.k_norm.weight"] = relay.shadow.k_norm.weight
