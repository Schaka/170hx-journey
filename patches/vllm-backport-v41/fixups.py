#!/usr/bin/env python3
"""Anchor-based fixes for the hunks of vLLM PR #56201 that `git apply` rejects
against the lazymio/vllm-backport sm80 tree. Run from the directory that holds
the `vllm/` package, after applying the PR diff with `git apply --reject`.
Every edit is idempotent."""
import sys

def sub(path, old, new, count=1, need=True, done_if=None):
    """Replace `old` with `new` in `path`, once, and do nothing on a re-run.

    `done_if` names a marker string. The upstream pull request lands some
    hunks on one merge base and rejects them on another. When the marker is
    already in the file, the change is in place in some form and this edit
    has nothing to do.
    """
    s = open(path).read()
    if new in s or (done_if is not None and done_if in s):
        print(f"skip  {path}"); return
    n = s.count(old)
    if n != count:
        msg = f"ANCHOR {path}: found {n}, wanted {count}"
        if need: raise SystemExit(msg)
        print("warn  " + msg); return
    open(path, "w").write(s.replace(old, new, count))
    print(f"ok    {path}")

# --- model / backend registration -----------------------------------------
sub("vllm/model_executor/models/registry.py",
    '    "Dots3NoteForCausalLM": (\n        "vllm.models.dots3_note",',
    '    "DeepseekV41ForCausalLM": (\n        "vllm.models.deepseek_v4_1",\n'
    '        "DeepseekV41ForCausalLM",\n    ),\n'
    '    "Dots3NoteForCausalLM": (\n        "vllm.models.dots3_note",')

sub("vllm/config/vllm.py",
    '        "Glm5NextForCausalLM",\n        "Glm5NextForConditionalGeneration",',
    '        "DeepseekV41ForCausalLM",\n'
    '        "Glm5NextForCausalLM",\n        "Glm5NextForConditionalGeneration",')

sub("vllm/config/model.py",
    '            elif arch in ("InklingForCausalLM", "InklingForConditionalGeneration"):',
    '            elif arch == "DeepseekV41ForCausalLM":\n'
    '                self.tokenizer_mode = "deepseek_v41"\n'
    '            elif arch in ("InklingForCausalLM", "InklingForConditionalGeneration"):')

sub("vllm/model_executor/models/config.py",
    '    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,',
    '    "DeepseekV41ForCausalLM": DeepseekV4ForCausalLMConfig,\n'
    '    "DeepseekV32ForCausalLM": DeepseekV32ForCausalLM,')

# The PR may or may not land this hunk, and it may carry other model names
# with it, so match on the set member rather than on the whole line.
_STR = "vllm/tool_parsers/structural_tag_registry.py"
_s = open(_STR).read()
if '"deepseek_v41"' in _s.split("SUPPORTED_STRUCTURAL_TAG_MODELS")[0]:
    print("skip  " + _STR)
else:
    sub(_STR,
        'VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset({"hermes", "kimi_k3"})',
        'VLLM_BUILTIN_STRUCTURAL_TAG_MODELS = frozenset(\n'
        '    {"deepseek_v41", "hermes", "kimi_k3"}\n)')

_bk_anchor = '''    TRITON_MLA_SPARSE_DSV4 = (
        "vllm.models.deepseek_v4.ampere.ampere_sparse.DeepseekV4AmpereMLASparseBackend"
    )
'''
sub("vllm/v1/attention/backends/registry.py", _bk_anchor, _bk_anchor + '''    # DeepSeek V4.1 sparse MLA backends (model-driven; selected via the V4.1
    # layer). Separate names from DSV4 so a V4.1 model never resolves the
    # V4.0 backend classes through this enum.
    FLASHMLA_SPARSE_DSV41 = (
        "vllm.models.deepseek_v4_1.sparse_mla.DeepseekV4FlashMLABackend"
    )
    FLASHINFER_MLA_SPARSE_DSV41 = (
        "vllm.models.deepseek_v4_1.nvidia.flashinfer_sparse."
        "DeepseekV4FlashInferMLASparseBackend"
    )
    ROCM_FLASHMLA_SPARSE_DSV41 = (
        "vllm.models.deepseek_v4_1.amd.rocm.DeepseekV4ROCMAiterMLASparseBackend"
    )
    TRITON_MLA_SPARSE_DSV41 = (
        "vllm.models.deepseek_v4_1.ampere.ampere_sparse."
        "DeepseekV41AmpereMLASparseBackend"
    )
''')

# --- hierarchical (two-level) sparse indexer -------------------------------
IDX = "vllm/model_executor/layers/sparse_attn_indexer.py"
a = """                        ops.top_k_per_row_prefill(
                            logits,
                            cu_seqlen_ks[r0:r1],
                            cu_seqlen_ke[r0:r1],
                            topk_indices[r0:r1],
                            r1 - r0,"""
sub(IDX, a, """                        if candidate_blocks is not None:
                            # Two-level selection (v4.1): the candidate source
                            # publishes its top blocks; later indexers mask
                            # their scores to them, both before the row top-k.
                            # This SM80 path is row-chunked, so slice the
                            # candidate rows the way the logits rows are.
                            chunk_candidates = candidate_blocks[
                                chunk.token_start + r0 : chunk.token_start + r1
                            ]
                            if candidate_write:
                                _select_candidate_blocks(
                                    logits,
                                    cu_seqlen_ks[r0:r1],
                                    cu_seqlen_ke[r0:r1],
                                    chunk_candidates.shape[1],
                                    candidate_block_size,
                                    chunk_candidates,
                                )
                            else:
                                _apply_candidate_mask(
                                    logits,
                                    cu_seqlen_ks[r0:r1],
                                    cu_seqlen_ke[r0:r1],
                                    chunk_candidates,
                                    candidate_block_size,
                                )
""" + a)

a = """                if logits.shape[0] > 0:
                    num_rows = logits.shape[0]
                    ops.top_k_per_row_prefill("""
sub(IDX, a, """                if logits.shape[0] > 0:
                    num_rows = logits.shape[0]
                    if candidate_blocks is not None:
                        chunk_candidates = candidate_blocks[
                            chunk.token_start : chunk.token_end
                        ]
                        if candidate_write:
                            _select_candidate_blocks(
                                logits,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                chunk_candidates.shape[1],
                                candidate_block_size,
                                chunk_candidates,
                            )
                        else:
                            _apply_candidate_mask(
                                logits,
                                cu_seqlen_ks,
                                cu_seqlen_ke,
                                chunk_candidates,
                                candidate_block_size,
                            )
                    ops.top_k_per_row_prefill(""")

a = """        num_rows = logits.shape[0]
        topk_indices = topk_indices_buffer[row_lo:row_hi, :topk_tokens]"""
sub(IDX, a, """        num_rows = logits.shape[0]
        if candidate_blocks is not None:
            # Two-level selection (v4.1) on the decode logits; columns are
            # request-local compressed positions. seq_lens is (B, next_n) for
            # native spec decode (per-row effective lens) and (B, 1) otherwise.
            vis = seq_lens.reshape(-1)
            row_repeat = next_n if vis.numel() != num_rows else 1
            vis = vis[:num_rows]
            decode_candidates = candidate_blocks[:num_rows]
            if candidate_write:
                _select_candidate_blocks(
                    logits,
                    None,
                    vis,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                    row_repeat,
                )
            else:
                _apply_candidate_mask(
                    logits,
                    None,
                    vis,
                    decode_candidates,
                    candidate_block_size,
                    row_repeat,
                )
        topk_indices = topk_indices_buffer[row_lo:row_hi, :topk_tokens]""")

# --- model runner ----------------------------------------------------------
MR = "vllm/v1/worker/gpu_model_runner.py"
a = """                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                if self.lora_config:"""
sub(MR, a, done_if="self.lookback_token_ids = self._make_buffer", new="""                self.model = model_loader.load_model(
                    vllm_config=self.vllm_config, model_config=self.model_config
                )
                lookback_depth = getattr(self.model, "token_lookback_depth", 0)
                if lookback_depth > 0:
                    self.lookback_token_ids = self._make_buffer(
                        self.max_num_reqs, lookback_depth, dtype=torch.int32
                    )
                if self.lora_config:""")

a = """            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {"""
sub(MR, a, a.replace("self._init_model_kwargs()", "self._init_model_kwargs(num_reqs=0)", 1))

# --- V4 MoE refactor completed by the PR -----------------------------------
V4 = "vllm/models/deepseek_v4/nvidia/model.py"
sub(V4, "        is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers",
        "        is_hash_moe = extract_layer_index(prefix) < num_hash_layers")
sub(V4, "                torch.empty(config.n_routed_experts, dtype=torch.float32),",
        "                torch.empty(self.n_routed_experts, dtype=torch.float32),",
        count=1, need=False)
print("fixups done")

# --- shims for subsystems the sm80 backport drops --------------------------
# lazymio's backport removes the DeepSeek V4 vision subsystem, so
# vllm/models/deepseek_v4/common/mm_preprocess.py is absent. The only symbol
# the PR's V4 MoE signature needs from it is the sentinel base id.
V4 = "vllm/models/deepseek_v4/nvidia/model.py"
_s = open(V4).read()
if "IMAGE_SENTINEL_BASE_ID" in _s and "IMAGE_SENTINEL_BASE_ID = " not in _s:
    _anchor = "\n\nclass DeepseekV4MoE(nn.Module):"
    assert _s.count(_anchor) == 1
    open(V4, "w").write(_s.replace(_anchor,
        "\n\n# Image tokens borrow five consecutive reserved in-vocab ids starting here.\n"
        "# The vision subsystem itself is not part of this sm80 build.\n"
        "IMAGE_SENTINEL_BASE_ID = 129257\n" + _anchor, 1))
    print("ok    " + V4 + " (IMAGE_SENTINEL_BASE_ID)")
else:
    print("skip  " + V4 + " (IMAGE_SENTINEL_BASE_ID)")

# Vendor the DeepSeek V4 vision files the backport drops. V4.1 is natively
# multimodal and imports them, so they must come back from upstream.
import os, shutil
_PR = os.environ.get("PR_TREE", "")
for rel in ("vllm/models/deepseek_v4/common/vision.py",
            "vllm/models/deepseek_v4/common/mm_preprocess.py"):
    if not os.path.exists(rel) and _PR and os.path.exists(os.path.join(_PR, rel)):
        shutil.copy2(os.path.join(_PR, rel), rel)
        print("ok    vendored " + rel)

# --- DeepSeek V4.1 dense-linear quantization -------------------------------
# The V4.1 model asks for `CkptCtx` and `ModelOptLinearMethod` from
# vllm.model_executor.layers.quantization.modelopt. Those come from an upstream
# refactor (QuantKeyScheme) that the sm80 backport does not carry. The backport
# still has the older per-format classes, and `ModelOptMxFp8LinearMethod` on top
# of Marlin already covers SM80. Add the two missing names as a thin layer over
# it, plus the one real behavior the PR adds: this checkpoint stores one MXFP8
# scale per block of 32 output rows, not one per row.
MO = "vllm/model_executor/layers/quantization/modelopt.py"
_s = open(MO).read()
if "class CkptCtx" in _s:
    print("skip  " + MO + " (V4.1 linear layer)")
else:
    _s += '''

# --- DeepSeek V4.1 compatibility -------------------------------------------
# Upstream replaced the per-format ModelOpt*LinearMethod classes with one
# ModelOptLinearMethod driven by a QuantSpec and a CkptCtx. This build keeps
# the older classes, so provide the two names the V4.1 model imports and route
# them to the MXFP8 path, which already runs on SM80 through Marlin.
from dataclasses import dataclass


@dataclass(frozen=True)
class CkptCtx:
    """Per-checkpoint facts a QuantKey cannot carry."""

    group_size: int | None = None
    scale_block_size: tuple[int, int] | None = None


class ModelOptBlockScaleMxFp8LinearMethod(ModelOptMxFp8LinearMethod):
    """MXFP8 linear whose checkpoint shares one scale over several rows.

    DeepSeek V4.1 writes `weight_block_size: [32, 32]`, so its scale tensor is
    [N/32, K/32] where vLLM's parameter is [N, K/32]. Expand the loaded tensor
    along the output dimension before the ordinary loader shards it.
    """

    def __init__(self, quant_config, block_rows: int) -> None:
        super().__init__(quant_config)
        self._block_rows = block_rows

    def process_weights_after_loading(self, layer) -> None:
        # DeepSeek V4.1's `wo_a` is the only MXFP8 linear whose weight is read
        # directly rather than through a GEMM: the SM8x output projection views
        # it as [groups, o_lora_rank, hidden] and runs its own einsum. Marlin
        # repacks four fp8 values into one int32, which makes that view fail
        # with "shape '[2, 1024, 4096]' is invalid for input of size 2097152".
        # `bmm_batch_size` marks those layers, so give them the emulation
        # kernel, which dequantizes to BF16 at load time and keeps the shape.
        # Then drop `weight_scale`, because the reader multiplies by it when it
        # is present and the weight is already dequantized.
        if getattr(layer, "bmm_batch_size", None) is not None:
            from vllm.model_executor.kernels.linear.mxfp8.emulation import (
                EmulationMxfp8LinearKernel,
            )
            from vllm.model_executor.kernels.linear.mxfp8.Mxfp8LinearKernel import (
                Mxfp8LinearLayerConfig,
            )

            self.kernel = EmulationMxfp8LinearKernel(
                Mxfp8LinearLayerConfig(bmm_batch_size=None)
            )
            super().process_weights_after_loading(layer)
            if layer.weight.element_size() >= 2:
                layer._parameters.pop("weight_scale", None)
                layer.weight_scale = None
            return
        super().process_weights_after_loading(layer)

    def create_weights(self, layer, *args, **kwargs) -> None:
        super().create_weights(layer, *args, **kwargs)
        if self._block_rows <= 1:
            return
        scale_param = layer.weight_scale
        base_loader = scale_param.weight_loader
        block_rows = self._block_rows

        def block_scale_loader(param, loaded_weight, *a, **k):
            loaded_weight = loaded_weight.view(torch.uint8).repeat_interleave(
                block_rows, dim=0
            )
            return base_loader(param, loaded_weight, *a, **k)

        scale_param.weight_loader = block_scale_loader


def ModelOptLinearMethod(spec, ctx=None):  # noqa: N802
    """Build a linear method from a QuantSpec, as upstream's class does."""
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Static,
    )

    if spec.weight is not kMxfp8Static:
        raise NotImplementedError(
            f"This build only maps kMxfp8Static onto ModelOptLinearMethod, "
            f"got {spec.weight}"
        )
    block_rows, block_cols = (ctx.scale_block_size if ctx else None) or (
        1,
        MXFP8_BLOCK_SIZE,
    )
    if block_cols != MXFP8_BLOCK_SIZE:
        raise NotImplementedError(
            f"MXFP8 checkpoint scale block {ctx.scale_block_size} is unsupported"
        )
    quant_config = ModelOptMxFp8Config(
        is_checkpoint_mxfp8_serialized=True,
        kv_cache_quant_algo=None,
        exclude_modules=[],
    )
    return ModelOptBlockScaleMxFp8LinearMethod(quant_config, block_rows)
'''
    open(MO, "w").write(_s)
    print("ok    " + MO + " (V4.1 linear layer)")

# --- route SM8x to the Ampere sparse-MLA shim ------------------------------
# `_select_dsv4_attn_cls` in the V4.1 model only knows FlashMLA and FlashInfer,
# neither of which has an SM80 kernel. Add the SM8x branch, the same shape as
# `vllm/models/deepseek_v4/nvidia/model.py` uses for V4.0.
V41 = "vllm/models/deepseek_v4_1/nvidia/model.py"
_s = open(V41).read()
_a = """    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):"""
if "DeepseekV41AmpereMLAAttention" in _s:
    print("skip  " + V41 + " (SM8x dispatch)")
else:
    assert _s.count(_a) == 1, f"anchor in {V41}: {_s.count(_a)}"
    _n = """    backend = vllm_config.attention_config.backend
    device_capability = current_platform.get_device_capability()
    if device_capability is not None and device_capability.major == 8:
        # SM8x has no fp8 hardware and no FlashMLA or FlashInfer sparse
        # kernels. The ROCm Triton sparse-MLA path is portable, so reuse it
        # the way vllm.models.deepseek_v4.ampere does for V4.0.
        if backend is not None and (
            backend != AttentionBackendEnum.TRITON_MLA_SPARSE_DSV41
        ):
            raise ValueError(
                f"{backend.name} is not supported for DeepSeek V4.1 on SM8x; "
                "use TRITON_MLA_SPARSE_DSV41 (default)."
            )
        if vllm_config.attention_config.use_fp4_indexer_cache:
            raise ValueError(
                "attention_config.use_fp4_indexer_cache requires SM100; "
                "the MXFP4 indexer kernels emit Blackwell-only PTX."
            )
        from vllm.models.deepseek_v4_1.ampere.ampere_sparse import (
            DeepseekV41AmpereMLAAttention,
        )

        return DeepseekV41AmpereMLAAttention
    if backend in (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
    ):"""
    open(V41, "w").write(_s.replace(_a, _n, 1))
    print("ok    " + V41 + " (SM8x dispatch)")

# --- the backport's indexer needs num_heads --------------------------------
# lazymio's SparseAttnIndexer takes a required `num_heads`, which drives its
# indexer decode-sharding path (VLLM_INDEXER_DECODE_SHARD_MIN_REQS). Upstream
# V4.1 does not pass it. The GLM-5.3 (deepseek_v32) call site in this same
# build passes `num_heads=self.n_head`, so do the same here.
sub("vllm/models/deepseek_v4_1/attention.py",
    """            self.topk_indices_buffer,
            skip_k_cache_insert=True,""",
    """            self.topk_indices_buffer,
            # Required by this build's indexer decode-sharding path.
            num_heads=self.n_head,
            skip_k_cache_insert=True,""")

# --- restore the vision routing bias on the MoE gate -----------------------
# The sm80 backport strips DeepSeek V4's vision code, and with it the
# `gate.bias_vl` parameter. V4.1 is natively multimodal and its checkpoint
# ships `layers.N.ffn.gate.bias_vl` for all 40 layers, so the loader raises
# `KeyError: 'layers.20.ffn.gate.bias_vl'` with nowhere to put it. The config
# adapter flattens `vision_config.num_hidden_layers` to `vision_n_layers`,
# which is 32 for this checkpoint, so restore the upstream block verbatim.
sub("vllm/models/deepseek_v4/nvidia/model.py",
    """        if config.n_shared_experts is None:
            self.shared_experts = None""",
    """        if getattr(config, "vision_n_layers", 0) > 0:
            # Vision checkpoints route image sentinel tokens with bias_vl
            # instead of e_score_correction_bias / the hash table. Created on
            # every MoE layer, hash layers included.
            self.gate.bias_vl = nn.Parameter(
                torch.empty(self.n_routed_experts, dtype=torch.float32),
                requires_grad=False,
            )

        if config.n_shared_experts is None:
            self.shared_experts = None""")

# --- name the layers the KV grouping drops ---------------------------------
# "Some layers are not assigned to any group." says nothing about which layers
# or which spec types. V4.1 carries five kinds of cache, so make the message
# list what was dropped.
_KVU = "vllm/v1/core/kv_cache_utils.py"
_old = (
    "        assert sum(len(group.layer_names) for group in projected_groups) == len(\n"
    "            kv_cache_spec_one_worker\n"
    "        ), \"Some layers are not assigned to any group.\""
)
_new = (
    "        _grouped = {\n"
    "            name for group in projected_groups for name in group.layer_names\n"
    "        }\n"
    "        _dropped = {\n"
    "            name: type(spec).__name__\n"
    "            for name, spec in kv_cache_spec_one_worker.items()\n"
    "            if name not in _grouped\n"
    "        }\n"
    "        assert not _dropped, (\n"
    "            \"Some layers are not assigned to any group. \"\n"
    "            f\"dropped={_dropped}. all spec types=\"\n"
    "            f\"{sorted({type(s).__name__ for s in kv_cache_spec_one_worker.values()})}\"\n"
    "        )"
)
sub(_KVU, _old, _new)

# --- group the compressor state caches -------------------------------------
# `group_and_unify_kv_cache_specs` buckets SlidingWindowMLASpec and
# MLAAttentionSpec and silently drops everything else. V4.1's compressor owns a
# CircularBufferSpec per kv-source layer, so those fall through the else and
# never reach a group, which trips "Some layers are not assigned to any group."
# Give them their own uniform group, the way the upstream state buckets do.
sub(_KVU,
    """    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            mla_specs[name] = spec""",
    """    circular_specs: dict[str, KVCacheSpec] = {}
    for name, spec in kv_cache_spec.items():
        if isinstance(spec, SlidingWindowMLASpec):
            grouped_swa_mla_specs[(spec.block_size, spec.sliding_window)][name] = spec
        elif type(spec) is CircularBufferSpec:
            # The V4.1 compressor state. Checked before MLAAttentionSpec so a
            # future subclass cannot fall into the wrong bucket.
            circular_specs[name] = spec
        elif isinstance(spec, MLAAttentionSpec):
            mla_specs[name] = spec""")

sub(_KVU,
    """    return [mla_uniform_spec, *swa_uniform_specs]""",
    """    circular_uniform_specs: list[UniformTypeKVCacheSpecs] = []
    if circular_specs:
        circular_uniform_spec = UniformTypeKVCacheSpecs.from_specs(circular_specs)
        assert circular_uniform_spec is not None
        circular_uniform_specs.append(circular_uniform_spec)

    return [mla_uniform_spec, *swa_uniform_specs, *circular_uniform_specs]""")

# --- carry the compressor-state group through the planner ------------------
# `_get_kv_cache_groups_uniform_groups` assumes `grouped_specs` is exactly
# [MLA, *SWA] and asserts every group after the first is SlidingWindowMLASpec.
# The compressor-state group added above is neither, so split it out before the
# layer-tuple arithmetic and pass it through as its own group.
sub(_KVU,
    """    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.""",
    """    # The compressor state is one circular buffer per kv-source layer. It has
    # no layer tuples to align, so keep it out of the padding arithmetic below
    # and hand it back as its own group.
    def _is_circular(group: UniformTypeKVCacheSpecs) -> bool:
        return any(
            type(spec) is CircularBufferSpec
            for spec in group.kv_cache_specs.values()
        )

    circular_groups = [
        KVCacheGroupSpec(
            layer_names=list(group.kv_cache_specs.keys()),
            kv_cache_spec=group,
        )
        for group in grouped_specs[1:]
        if _is_circular(group)
    ]
    grouped_specs = [grouped_specs[0]] + [
        group for group in grouped_specs[1:] if not _is_circular(group)
    ]

    # We define a layer tuple as a group of layers with different page sizes, and
    # one UniformTypeKVCacheSpecs contains a list of layer tuples.""")

sub(_KVU,
    """    return [full_mla_group, *swa_mla_groups]""",
    """    return [full_mla_group, *swa_mla_groups, *circular_groups]""")

# --- skip KV tensors for layers this worker does not own -------------------
# `_project_kv_cache_groups_to_worker` keeps every global group, and when a
# group projects to no layer on this stage it leaves the group carrying the
# global spec dict. The tensor builder then emits tensors for layers that live
# on another pipeline stage, and `allocate_kv_cache` dies on a bare
# `next(...)` with an empty StopIteration. V4.1 hits this with the compressor
# state caches, which sit only on the stage that owns their source layer.
# Skip those tensors: this worker has no layer to map them to, so it never
# reads them. A layer that is genuinely missing still fails later, loudly,
# when the model asks for its cache.
sub("vllm/v1/worker/utils.py",
    """        group_id, group = next(
            (group_id, group)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if layer_name in group.layer_names
        )""",
    """        found = next(
            (
                (group_id, group)
                for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
                if layer_name in group.layer_names
            ),
            None,
        )
        if found is None:
            logger.debug_once(
                "Skipping KV cache tensor for %s: no group on this worker.",
                layer_name,
            )
            continue
        group_id, group = found""")

# --- Engram lookup: decode e4m3 in software on SM8x ------------------------
# Triton types a kernel parameter from the dtype of the tensor it receives, so
# a float8_e4m3fn table makes `_engram_lookup_kernel` declare fp8e4nv, which
# Triton refuses below SM89: "type fp8e4nv not supported in this
# architecture". No runtime branch has to reach the cast. Pass the table as
# uint8 and decode with the integer path this build already ships for the
# sparse-MLA kernels. Both dtypes are one byte, so the view costs nothing.
_ENG = "vllm/models/deepseek_v4_1/common/engram.py"
sub(_ENG,
    """        weight, scales = self._storage()""",
    """        weight, scales = self._storage()
        # SM8x cannot name fp8e4nv in a Triton signature. See
        # _engram_lookup_kernel, which decodes the byte itself.
        if weight.dtype != torch.uint8:
            weight = weight.view(torch.uint8)""")

sub(_ENG,
    """        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0.0,
        )""",
    """        values = tl.load(
            weight + local[:, None] * DIM + cols[None, :],
            mask=owned[:, None],
            other=0,
        )""")

sub(_ENG,
    """            (values.to(tl.float32) * scale).to(tl.bfloat16),""",
    """            (_e4m3fn_to_f32_alu(values) * scale).to(tl.bfloat16),""")

sub(_ENG,
    """def _engram_lookup_kernel(""",
    """def _engram_lookup_kernel(  # noqa: E302""")

_s = open(_ENG).read()
if "from vllm.v1.attention.ops.fp8_sm80 import" not in _s:
    _anchor = "from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor"
    assert _s.count(_anchor) == 1
    open(_ENG, "w").write(
        _s.replace(
            _anchor,
            _anchor + "\nfrom vllm.v1.attention.ops.fp8_sm80 import _e4m3fn_to_f32_alu",
            1,
        )
    )
    print("ok    " + _ENG + " (fp8_sm80 import)")

# --- route the fused Q/KV prologue to Triton on SM8x -----------------------
# The compiled `_C` op in this wheel is the V4.0 9-argument version, which
# always applies the weightless Q RMSNorm. V4.1 passes a tenth argument,
# `apply_q_norm=False`, and the call fails with "expected at most 9
# argument(s) but received 10". Dropping the argument would norm Q twice
# instead, so use the Triton stand-in for SM8x.
sub("vllm/models/deepseek_v4_1/attention.py",
    """            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(""",
    """            swa_kv_cache_2d = swa_kv_cache.view(swa_kv_cache.shape[0], -1)
            from vllm.platforms import current_platform as _plat

            _cap = _plat.get_device_capability()
            if _cap is not None and _cap.major == 8:
                from vllm.models.deepseek_v4_1.ampere.qnorm_rope_kv_insert import (
                    qnorm_rope_kv_quant_insert,
                )

                return qnorm_rope_kv_quant_insert(
                    q,
                    kv,
                    swa_kv_cache_2d,
                    swa_metadata.slot_mapping,
                    positions,
                    cos_sin_cache,
                    self.padded_heads,
                    self.eps,
                    swa_metadata.block_size,
                    False,
                )
            return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(""")

# --- cache_utils: encode and decode fp8 in software on SM8x ----------------
# `quantize_and_insert_k_cache` casts bf16 to `tl.float8e4nv`, and the dequant
# path bitcasts back. Triton refuses that type below SM89. This build already
# ships FNUZ-aware software encode and decode for the GLM-5.3 sparse-MLA
# kernels, and both produce the identical byte, so route through them.
_CU = "vllm/models/deepseek_v4_1/common/ops/cache_utils.py"
sub(_CU,
    """            # Convert to fp8 (FNUZ on gfx942, OCP elsewhere), then bitcast to uint8.
            if use_fnuz:
                x_fp8 = x_clamped.to(tl.float8e4b8)
            else:
                x_fp8 = x_clamped.to(tl.float8e4nv)
            x_uint8 = x_fp8.to(tl.uint8, bitcast=True)""",
    """            # Convert to fp8 (FNUZ on gfx942, OCP elsewhere) as a byte. The
            # helper picks the native cast where the architecture has one and
            # an integer encoder below SM89, which Triton needs because it
            # types the value from the tensor dtype.
            x_uint8 = _encode_fp8_u8(x_clamped, use_fnuz)""")

sub(_CU,
    """                # Bitcast uint8 back to fp8 (FNUZ on gfx942, OCP elsewhere).
                if use_fnuz:
                    x_fp8 = x_uint8.to(tl.float8e4b8, bitcast=True)
                else:
                    x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)

                # Convert fp8 to float32 for computation
                x_float = x_fp8.to(tl.float32)""",
    """                # Byte back to float32. Same helper choice as the encode
                # above: native bitcast where it exists, integer math below
                # SM89.
                x_float = _decode_fp8_f32(x_uint8, use_fnuz)""")

_s = open(_CU).read()
if "from vllm.v1.attention.ops.fp8_sm80 import" not in _s:
    for _cand in (
        "from vllm.triton_utils import tl, triton\n",
        "from vllm.triton_utils import triton, tl\n",
    ):
        if _cand in _s:
            open(_CU, "w").write(
                _s.replace(
                    _cand,
                    _cand
                    + "from vllm.v1.attention.ops.fp8_sm80 import (\n"
                    + "    _decode_fp8_f32,\n"
                    + "    _encode_fp8_u8,\n"
                    + ")\n",
                    1,
                )
            )
            print("ok    " + _CU + " (fp8_sm80 import)")
            break
    else:
        raise SystemExit("ANCHOR " + _CU + ": no triton_utils import found")

# --- two more byte-producing fp8 casts on the V4.1 path --------------------
# Same rule as cache_utils: the value is stored as a byte, so encode it as one
# instead of naming a type Triton rejects below SM89.
def _add_fp8_sm80_import(path, names):
    src = open(path).read()
    if "vllm.v1.attention.ops.fp8_sm80" in src:
        print("skip  " + path + " (fp8_sm80 import)")
        return
    for cand in (
        "from vllm.triton_utils import tl, triton\n",
        "from vllm.triton_utils import triton, tl\n",
    ):
        if cand in src:
            open(path, "w").write(
                src.replace(
                    cand,
                    cand
                    + "from vllm.v1.attention.ops.fp8_sm80 import "
                    + ", ".join(names)
                    + "\n",
                    1,
                )
            )
            print("ok    " + path + " (fp8_sm80 import)")
            return
    raise SystemExit("ANCHOR " + path + ": no triton_utils import found")


_FCQ = "vllm/models/deepseek_v4_1/common/ops/fused_compress_quant_cache.py"
sub(_FCQ,
    """    fp8 = tl.clamp(scaled, -448.0, 448.0).to(tl.float8e4nv)
    packed = tl.reshape(fp8.to(tl.uint8, bitcast=True), (512,))""",
    """    packed = tl.reshape(
        _encode_e4m3fn_u8(tl.clamp(scaled, -448.0, 448.0)), (512,)
    )""")
_add_fp8_sm80_import(_FCQ, ["_encode_e4m3fn_u8"])

_IKS = "vllm/models/deepseek_v4_1/common/ops/indexer_k_store.py"
sub(_IKS,
    """        x_uint8 = x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)""",
    """        x_uint8 = _encode_e4m3fn_u8(x_clamped)""")
_add_fp8_sm80_import(_IKS, ["_encode_e4m3fn_u8"])

# --- gate the CuteDSL dequant-gather on capability, not package presence ---
# `dequantize_and_gather_k_cache` dispatches on `has_cutedsl()`, which only
# asks whether the package is installed. The CuteDSL kernels target SM90+, so
# on sm_80 the dispatch is taken and then libNVVM aborts with "NVVM backend
# compilation failed ... target architecture: sm_80". This build already has
# the right predicate, `is_cutedsl_supported()`, which adds the capability
# check for exactly this reason. There is a Triton implementation right below.
sub(_CU,
    """    if has_cutedsl():""",
    """    if is_cutedsl_supported():""")

_s = open(_CU).read()
if "is_cutedsl_supported" not in _s.split("def dequantize_and_gather_k_cache")[0]:
    for _cand in (
        "from vllm.utils.import_utils import has_cutedsl\n",
        "from vllm.utils.import_utils import has_cutedsl, is_cutedsl_supported\n",
    ):
        if _cand in _s:
            open(_CU, "w").write(
                _s.replace(
                    _cand,
                    "from vllm.utils.import_utils import (\n"
                    "    has_cutedsl,\n"
                    "    is_cutedsl_supported,\n"
                    ")\n",
                    1,
                )
            )
            print("ok    " + _CU + " (is_cutedsl_supported import)")
            break
    else:
        raise SystemExit("ANCHOR " + _CU + ": has_cutedsl import not found")

# --- DSpark under pipeline parallel ---------------------------------------
# The DSpark drafter runs on the last pipeline stage and owns no embedding
# table. It aliases the target model table. Under PP the target only builds
# that table on the first stage, so the alias finds a PPMissingLayer and
# load_dspark_model raises. Build the table on the last stage too when a
# DSpark drafter is configured. The weight loader keys off the parameter
# existing, through is_pp_missing_parameter, so the weights arrive by
# themselves. The cost is one extra 129280 x 5120 table, split over the
# tensor-parallel ranks of that stage.
_MODEL = "vllm/models/deepseek_v4_1/nvidia/model.py"
sub(_MODEL,
    """        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(""",
    """        _spec = vllm_config.speculative_config
        _draft_needs_embed = (
            get_pp_group().is_last_rank
            and _spec is not None
            and getattr(_spec, "method", None) == "dspark"
        )
        if get_pp_group().is_first_rank or _draft_needs_embed:
            self.embed_tokens = VocabParallelEmbedding(""")

# --- allow a pipeline cut inside a kv-sharing group ------------------------
# Layers 20 to 39 read one compressed KV cache and one indexer K cache that
# layer 20 writes. Stock vLLM refuses a pipeline cut inside that range, which
# pins those 20 layers to one stage and forces 8 GPUs. The relay module gives
# a later stage its own copy of both caches and refills them from the source
# latent, which rides the pipeline hop. See ampere/pp_kv_group_relay.py.
_ATT = "vllm/models/deepseek_v4_1/attention.py"

# 1. The indexer K cache. Build a local one instead of refusing.
sub(_ATT,
    """                index_k_cache = self._static_forward_context.get(k_cache_prefix)
                if index_k_cache is None:
                    raise NotImplementedError(
                        f"Indexer K cache source {k_cache_prefix} not found on "
                        "this rank; PP splits inside a v4.1 kv-sharing group "
                        "are not supported."
                    )""",
    """                index_k_cache = self._static_forward_context.get(k_cache_prefix)
                if index_k_cache is None:
                    # The source sits on an earlier pipeline stage. Own a
                    # copy here; the relay refills it every step.
                    index_k_cache = DeepseekV4IndexerCache(
                        head_dim=_indexer_k_cache_head_dim(
                            config.index_head_dim, dsa_indexer_uses_fp4(vllm_config)
                        ),
                        dtype=torch.uint8,
                        prefix=k_cache_prefix,
                        cache_config=cache_config,
                        compress_ratio=self.compress_ratio,
                    )
                    self._relay_index_k_cache = index_k_cache""")

# 2. The compressed KV cache. Build the replica and the relay.
sub(_ATT,
    """            if (
                not self.is_kv_source
                and self.compressed_cache_prefix not in self._static_forward_context
            ):
                raise NotImplementedError(
                    f"Compressed-KV source {self.compressed_cache_prefix} not "
                    "found on this rank; PP splits inside a v4.1 kv-sharing "
                    "group are not supported."
                )""",
    """            if (
                not self.is_kv_source
                and self.compressed_cache_prefix not in self._static_forward_context
            ):
                from vllm.models.deepseek_v4_1.pp_kv_group_relay import build_relay

                # The source sits on an earlier pipeline stage. The first
                # consumer on this stage builds the replica set, and every
                # later consumer resolves to it through the forward context.
                build_relay(
                    consumer_attn=self,
                    source_attn_prefix=self.compressed_cache_prefix,
                    index_k_cache=getattr(self, "_relay_index_k_cache", None),
                    cache_config=cache_config,
                    config=config,
                )""")

# 3. The source layer publishes its latent for the next stage.
# The copy sits after both parallel blocks join. The compressor runs on an
# auxiliary stream, so a copy placed earlier can read the latent before that
# stream has written it.
sub(_ATT,
    """        index_q, index_q_scale, index_weights_out = indexer_result""",
    """        index_q, index_q_scale, index_weights_out = indexer_result

        if latent is not None and self._relay_latent_buffer is not None:
            # The next pipeline stage rebuilds both cache writes from this.
            self._relay_latent_buffer[: latent.shape[0]].copy_(latent)""")

# The buffer defaults to None, so a layer that publishes nothing costs nothing.
sub(_ATT,
    """        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer""",
    """        self.topk_indices_buffer = topk_indices_buffer
        self.candidate_block_buffer = candidate_block_buffer
        # Set by the model on the one kv-source layer whose group continues
        # onto the next pipeline stage.
        self._relay_latent_buffer: torch.Tensor | None = None""")
print("relay fixups done")

_M41 = "vllm/models/deepseek_v4_1/nvidia/model.py"

# 4. Collect the relays the layers built, and give the last local kv source a
# buffer to publish its latent into.
sub(_M41,
    """        # The n-gram hash needs a slot-keyed rolling store of compressed ids""",
    """        from vllm.models.deepseek_v4_1.pp_kv_group_relay import take_relays

        # A pipeline cut inside a kv-sharing group leaves readers on this
        # stage without their writer. The consumer layers built one relay per
        # split group. Register them so their weights load and move to device.
        self.kv_group_relays = nn.ModuleList(take_relays())

        # Every stage that sends carries the latent slot, whether or not it
        # fills it. A stage that never fills it still keeps the pipeline
        # payload the same shape on both sides of the hop.
        self._relay_latent_buffer: torch.Tensor | None = None
        self._relays_forward_latent = False
        if get_pp_group().world_size > 1 and not get_pp_group().is_last_rank:
            # Zeroed, not empty. A step where the source produces no latent
            # leaves these rows untouched, and the receiving stage still
            # writes them. Uninitialized memory would put NaN in the cache.
            self._relay_latent_buffer = torch.zeros(
                vllm_config.scheduler_config.max_num_batched_tokens,
                config.head_dim,
                dtype=vllm_config.model_config.dtype,
            )
            _local_sources = [
                layer
                for layer in islice(self.layers, self.start_layer, self.end_layer)
                if isinstance(layer, DeepseekV4DecoderLayer)
                and getattr(layer.attn, "is_kv_source", False)
            ]
            if _local_sources:
                # Only the last one can own a group that continues onward.
                _local_sources[-1].attn._relay_latent_buffer = (
                    self._relay_latent_buffer
                )
            else:
                # This stage writes no kv source at all, so the group that
                # straddles it starts further back. Pass on what arrives, or
                # the next stage reads a buffer of zeros.
                self._relays_forward_latent = True

        # The n-gram hash needs a slot-keyed rolling store of compressed ids""")

# 5. Carry the latent across the pipeline hop.
sub(_M41,
    """                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
            }
        )""",
    """                "pre_mix": torch.zeros(
                    (batch_size, self.hc_mult),
                    dtype=torch.float32,
                    device=device,
                ),
                # The compressor latent of the last kv source on the sending
                # stage. Refills the replicated caches of a split group.
                "kv_latent": torch.zeros(
                    (batch_size, self.config.head_dim),
                    dtype=dtype,
                    device=device,
                ),
            }
        )""")

sub(_M41,
    """        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "pre_mix": pre_mix}
            )""",
    """        if not get_pp_group().is_last_rank:
            assert self._relay_latent_buffer is not None
            return IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "pre_mix": pre_mix,
                    "kv_latent": self._relay_latent_buffer[: positions.shape[0]],
                }
            )""")

# 6. Refill the replicated caches before any consumer layer reads them.
sub(_M41,
    """        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]""",
    """        if not get_pp_group().is_first_rank:
            assert intermediate_tensors is not None
            pre_mix = intermediate_tensors["pre_mix"]
            for _relay in self.kv_group_relays:
                _relay.write(
                    intermediate_tensors["kv_latent"][: positions.shape[0]],
                    positions,
                )
            if "topk_indices" in intermediate_tensors.tensors:
                _n = positions.shape[0]
                self.topk_indices_buffer[:_n].copy_(
                    intermediate_tensors["topk_indices"][:_n]
                )
            if self._relays_forward_latent:
                _n = positions.shape[0]
                self._relay_latent_buffer[:_n].copy_(
                    intermediate_tensors["kv_latent"][:_n]
                )""")

# 7. Let the shadow weights load under their source-layer checkpoint names.
sub(_M41,
    """        params_dict = dict(self.named_parameters())""",
    """        params_dict = dict(self.named_parameters())
        if self.kv_group_relays:
            from vllm.models.deepseek_v4_1.pp_kv_group_relay import (
                alias_source_params,
            )

            alias_source_params(list(self.kv_group_relays), params_dict)""")
print("relay model fixups done")

# 8. The source layer is a PPMissingLayer on a stage that only reads its
# group, so is_pp_missing_parameter reports the shadow weights as absent and
# the loader skips them. Uninitialized index keys make the sparse indexer
# select the wrong tokens, which reads as fluent text about the wrong part of
# the context. Consult the relay alias names before trusting that report.
sub(_M41,
    """        params_dict = dict(self.named_parameters())
        if self.kv_group_relays:
            from vllm.models.deepseek_v4_1.pp_kv_group_relay import (
                alias_source_params,
            )

            alias_source_params(list(self.kv_group_relays), params_dict)""",
    """        params_dict = dict(self.named_parameters())
        _relay_aliases: set[str] = set()
        if self.kv_group_relays:
            from vllm.models.deepseek_v4_1.pp_kv_group_relay import (
                alias_names,
                alias_source_params,
            )

            alias_source_params(list(self.kv_group_relays), params_dict)
            _relay_aliases = alias_names(list(self.kv_group_relays))

        def _pp_missing(param_name: str) -> bool:
            if param_name in _relay_aliases:
                return False
            return is_pp_missing_parameter(param_name, self)""")

sub(_M41,
    """                if is_pp_missing_parameter(name, self):
                    break""",
    """                if _pp_missing(name):
                    break""")

sub(_M41,
    """                        if is_pp_missing_parameter(name_mapped, self):
                            continue""",
    """                        if _pp_missing(name_mapped):
                            continue""")

sub(_M41,
    """                elif "attn_sink" in name:
                    if is_pp_missing_parameter(name, self):
                        continue""",
    """                elif "attn_sink" in name:
                    if _pp_missing(name):
                        continue""")

sub(_M41,
    """                else:
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at""",
    """                else:
                    if _pp_missing(name):
                        continue
                    # Non-LoRA params on a LoRA-wrapped module live at""")
print("relay weight-loading fixups done")



# --- relay the candidate blocks too ----------------------------------------
# Layer 20 is the candidate source. It publishes its top candidate blocks into
# a buffer that every later indexer masks its scores against. A stage that
# holds those later layers but not layer 20 has an empty buffer, so its
# indexers select the wrong tokens. Carry the buffer across the hop.
sub(_M41,
    """                # The compressor latent of the last kv source on the sending
                # stage. Refills the replicated caches of a split group.
                "kv_latent": torch.zeros(
                    (batch_size, self.config.head_dim),
                    dtype=dtype,
                    device=device,
                ),
            }
        )""",
    """                # The compressor latent of the last kv source on the sending
                # stage. Refills the replicated caches of a split group.
                "kv_latent": torch.zeros(
                    (batch_size, self.config.head_dim),
                    dtype=dtype,
                    device=device,
                ),
                # The top-k indices that the last index source on the
                # sending stage published. A stage whose first layers run no
                # indexer of their own read them from here.
                "topk_indices": torch.zeros(
                    (batch_size, self.topk_indices_buffer.shape[1]),
                    dtype=torch.int32,
                    device=device,
                ),
                **(
                    {
                        "candidate_blocks": torch.zeros(
                            (batch_size, self.candidate_block_buffer.shape[1]),
                            dtype=torch.int32,
                            device=device,
                        )
                    }
                    if self.candidate_block_buffer is not None
                    else {}
                ),
            }
        )""")

sub(_M41,
    """                    "kv_latent": self._relay_latent_buffer[: positions.shape[0]],
                }
            )""",
    """                    "kv_latent": self._relay_latent_buffer[: positions.shape[0]],
                    "topk_indices": self.topk_indices_buffer[
                        : positions.shape[0]
                    ],
                    **(
                        {
                            "candidate_blocks": self.candidate_block_buffer[
                                : positions.shape[0]
                            ]
                        }
                        if self.candidate_block_buffer is not None
                        else {}
                    ),
                }
            )""")

sub(_M41,
    """            for _relay in self.kv_group_relays:
                _relay.write(""",
    """            if (
                self.candidate_block_buffer is not None
                and "candidate_blocks" in intermediate_tensors.tensors
            ):
                _n = positions.shape[0]
                self.candidate_block_buffer[:_n].copy_(
                    intermediate_tensors["candidate_blocks"][:_n]
                )
            for _relay in self.kv_group_relays:
                _relay.write(""")
print("candidate relay done")

# --- emit reasoning under both field names --------------------------------
# vLLM names the reasoning field `reasoning`. DeepSeek's own API and most
# other providers name it `reasoning_content`, and that is the name agent
# clients look for, opencode included. A client that finds neither treats the
# whole thinking block as assistant content, writes it back into the next
# turn, and the model then drifts into a repetition loop. Emit both names.
_ALIAS_OLD = """    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if len(data.get("tool_calls", [])) == 0:
            data.pop("tool_calls", None)
        return data"""
_ALIAS_NEW = """    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if len(data.get("tool_calls", [])) == 0:
            data.pop("tool_calls", None)
        if data.get("reasoning") is not None:
            # Alias, not a move. A client that reads either name works.
            data["reasoning_content"] = data["reasoning"]
        return data"""

# The complete message, for a request that does not stream.
sub("vllm/entrypoints/openai/chat_completion/protocol.py", _ALIAS_OLD, _ALIAS_NEW)
# Every streamed delta. A client that checks each chunk needs it here too.
sub("vllm/entrypoints/openai/engine/protocol.py", _ALIAS_OLD, _ALIAS_NEW)
print("reasoning_content alias done")

# --- parser: recover a tool call the model opens inside <think> ------------
# The parser starts a turn in REASONING, because V4.1 means thinking when the
# request omits the thinking flag. The tree recovers a tool call that lost its
# envelope only from CONTENT, so a call that opens before `</think>` never
# reaches the recovery path. It leaks as raw protocol text inside the thinking
# block, and the block never closes cleanly.
_PARSER = "vllm/parser/deepseek_v4.py"

sub(_PARSER,
    """            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                validate_tool_name=True,
            ),
""",
    """            (ParserState.CONTENT, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                validate_tool_name=True,
            ),
            # The same recovery, reached while still inside <think>. Close
            # the reasoning block first, so the text before the marker stays
            # reasoning and the tool call does not land in it.
            (ParserState.REASONING, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
                validate_tool_name=True,
            ),
            (ParserState.REASONING, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_BLOCK,
                (EventType.REASONING_END, EventType.TEXT_CHUNK),
            ),
""")

# --- parser: accept a near-miss tool-calls envelope ------------------------
# Near the context ceiling the checkpoint sometimes writes the envelope with
# an underscore in place of the space, around a well-formed body. The lexer
# matches the longest literal first, so the correct envelope is untouched.
sub(_PARSER,
    """DSML_FOREIGN_TOOL_END = f"</{_DSML}function_calls>"
""",
    """DSML_FOREIGN_TOOL_END = f"</{_DSML}function_calls>"
# Near-miss envelope spellings, accepted so the call parses instead of
# reaching the client as raw protocol text
DSML_TOOL_START_LENIENT = f"<{_DSML}_tool_calls>"
DSML_TOOL_END_LENIENT = f"</{_DSML}_tool_calls>"
""")

sub(_PARSER,
    """            "TOOL_END": DSML_TOOL_END,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
""",
    """            "TOOL_END": DSML_TOOL_END,
            "TOOL_START_LENIENT": DSML_TOOL_START_LENIENT,
            "TOOL_END_LENIENT": DSML_TOOL_END_LENIENT,
            "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
""")

sub(_PARSER,
    """            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
""",
    """            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
            (ParserState.REASONING, "TOOL_START_LENIENT"): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START_LENIENT"): Transition(
                ParserState.TOOL_PREAMBLE,
                (),
            ),
""")

sub(_PARSER,
    """            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
""",
    """            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END_LENIENT"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
""")

sub(_PARSER,
    """            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
""",
    """            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END_LENIENT"): Transition(
                ParserState.CONTENT,
                (),
            ),
""")

# V4.1 spaces its markers, so it needs its own near-miss spelling. Its block
# name is " calls", and the near miss drops the space. The transitions above
# carry over, because the V4.1 configuration only replaces the literals.
sub("vllm/parser/deepseek_v41.py",
    """DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"
""",
    """DSML_PARAM_CLOSE = "</｜DSML｜ parameter>"
DSML_TOOL_START_LENIENT = "<｜DSML｜calls>"
DSML_TOOL_END_LENIENT = "</｜DSML｜calls>"
""")

sub("vllm/parser/deepseek_v41.py",
    """        "TOOL_END": DSML_TOOL_END,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
""",
    """        "TOOL_END": DSML_TOOL_END,
        "TOOL_START_LENIENT": DSML_TOOL_START_LENIENT,
        "TOOL_END_LENIENT": DSML_TOOL_END_LENIENT,
        "INVOKE_PREFIX": DSML_INVOKE_PREFIX,
""")
print("parser reasoning-state recovery and lenient envelope done")

# --- parser: accept a plain-ASCII tool-call envelope -----------------------
# `｜DSML｜` is one vocab token. The checkpoint sometimes drops it and writes
# `<invoke name="...">` in plain text, and it mixes the two spellings inside
# one call: an ASCII opener with a `</｜DSML｜ invoke>` close. The ASCII
# opener matches no terminal, so the parser never enters tool mode and the
# whole call leaks as text, which also leaves the thinking block open.
#
# Only the invoke marker gets a CONTENT and REASONING entry, and both
# validate the tool name against the request. An ASCII `<tool_calls>` wrapper
# stays plain text on purpose: a bare wrapper carries no name to check, and
# prose that talks about the protocol would then eat the rest of the message.
sub(_PARSER,
    """DSML_TOOL_END_LENIENT = f"</{_DSML}_tool_calls>"
""",
    """DSML_TOOL_END_LENIENT = f"</{_DSML}_tool_calls>"
# The same markers with the special token dropped. The checkpoint mixes
# these with the real spelling inside one call.
DSML_INVOKE_PREFIX_ASCII = '<invoke name="'
DSML_INVOKE_END_ASCII = "</invoke>"
DSML_TOOL_END_ASCII = "</tool_calls>"
""")

sub(_PARSER,
    """            "TOOL_START_LENIENT": DSML_TOOL_START_LENIENT,
            "TOOL_END_LENIENT": DSML_TOOL_END_LENIENT,
""",
    """            "TOOL_START_LENIENT": DSML_TOOL_START_LENIENT,
            "TOOL_END_LENIENT": DSML_TOOL_END_LENIENT,
            "INVOKE_PREFIX_ASCII": DSML_INVOKE_PREFIX_ASCII,
            "INVOKE_END_ASCII": DSML_INVOKE_END_ASCII,
            "TOOL_END_ASCII": DSML_TOOL_END_ASCII,
""")

sub(_PARSER,
    """            (ParserState.REASONING, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_BLOCK,
                (EventType.REASONING_END, EventType.TEXT_CHUNK),
            ),
""",
    """            (ParserState.REASONING, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_BLOCK,
                (EventType.REASONING_END, EventType.TEXT_CHUNK),
            ),
            # The same two recoveries for the plain-ASCII invoke marker.
            (ParserState.CONTENT, "INVOKE_PREFIX_ASCII"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
                validate_tool_name=True,
            ),
            (ParserState.REASONING, "INVOKE_PREFIX_ASCII"): Transition(
                ParserState.TOOL_NAME,
                (EventType.REASONING_END, EventType.TOOL_CALL_START),
                validate_tool_name=True,
            ),
""")

sub(_PARSER,
    """            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
""",
    """            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_PREAMBLE, "INVOKE_PREFIX_ASCII"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
""")

sub(_PARSER,
    """            (ParserState.TOOL_ARGS, "TOOL_END_LENIENT"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
""",
    """            (ParserState.TOOL_ARGS, "TOOL_END_LENIENT"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "INVOKE_END_ASCII"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END_ASCII"): Transition(
                ParserState.CONTENT,
                (EventType.TOOL_CALL_END,),
            ),
""")

sub(_PARSER,
    """            (ParserState.TOOL_BETWEEN, "TOOL_END_LENIENT"): Transition(
                ParserState.CONTENT,
                (),
            ),
""",
    """            (ParserState.TOOL_BETWEEN, "TOOL_END_LENIENT"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "INVOKE_PREFIX_ASCII"): Transition(
                ParserState.TOOL_NAME,
                (EventType.TOOL_CALL_START,),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END_ASCII"): Transition(
                ParserState.CONTENT,
                (),
            ),
""")

# The parameter markers need the same tolerance, and the ASCII spelling also
# drops the `string=` attribute. Each side of a parameter is matched on its
# own, because the checkpoint opens in one spelling and closes in the other.
sub(_PARSER,
    """_ESCAPED_DSML = re.escape(_DSML)
_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\\s+name="([^"]+)"\\s+string="(true|false)">'
    rf"(.*?)</{_ESCAPED_DSML}parameter>",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    rf'<{_ESCAPED_DSML}parameter\\s+name="([^"]+)"\\s+string="(true|false)">'
    rf"(.*)$",
    re.DOTALL,
)
""",
    """_ESCAPED_DSML = re.escape(_DSML)
_PARAM_OPEN = (
    rf'<(?:{_ESCAPED_DSML})?parameter\\s+name="([^"]+)"'
    rf'(?:\\s+string="(true|false)")?>'
)
_PARAM_SHUT = rf"</(?:{_ESCAPED_DSML})?parameter>"
_PARAM_RE = re.compile(
    _PARAM_OPEN + r"(.*?)" + _PARAM_SHUT,
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    _PARAM_OPEN + r"(.*)$",
    re.DOTALL,
)
""")

# V4.1 spaces its own markers, so it carries its own pair of patterns.
sub("vllm/parser/deepseek_v41.py",
    """_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\\s+name="([^"]+)"\\s+string="(true|false)">'
    r"(.*?)"
    r"(?:</｜DSML｜ parameter>|(?=<｜DSML｜ parameter\\s+name=))",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    r'<｜DSML｜ parameter\\s+name="([^"]+)"\\s+string="(true|false)">'
    r"(.*)$",
    re.DOTALL,
)
""",
    """_PARAM_OPEN = (
    r'<(?:｜DSML｜\\s*)?parameter\\s+name="([^"]+)"'
    r'(?:\\s+string="(true|false)")?>'
)
_PARAM_SHUT = r"</(?:｜DSML｜\\s*)?parameter>"
_PARAM_NEXT = r'(?=<(?:｜DSML｜\\s*)?parameter\\s+name=)'
_PARAM_RE = re.compile(
    _PARAM_OPEN + r"(.*?)" + r"(?:" + _PARAM_SHUT + r"|" + _PARAM_NEXT + r")",
    re.DOTALL,
)
_PARTIAL_PARAM_RE = re.compile(
    _PARAM_OPEN + r"(.*)$",
    re.DOTALL,
)
""")
print("parser plain-ASCII envelope done")
