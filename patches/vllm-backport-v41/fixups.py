#!/usr/bin/env python3
"""Anchor-based edits that add the missing pieces to the vllm-backport sm80
image for DeepSeek-V4.1-Flash on this box. Run from the directory that holds
the `vllm/` package. Every edit is idempotent.

The base image already carries the V4.1 model, the Ampere sparse-MLA backend
and the software fp8 helpers that SM8x needs. Four things are left:

1. The DSpark drafter needs an embedding table on the last pipeline stage.
2. A pipeline cut inside a kv-sharing group. See pp_kv_group_relay.py.
3. The chat API emits reasoning text under the name `reasoning` only.
4. The parser recovers a lost tool-call envelope from the content state only.
5. The DSpark aux hidden states do not cross a pipeline hop.
6. The cache allocator needs every group to hold a layer on this stage.
"""

def sub(path, old, new, count=1, need=True, done_if=None):
    """Replace `old` with `new` in `path`, once, and do nothing on a re-run.

    `done_if` names a marker string. When the marker is already in the file,
    the change is in place in some form and this edit has nothing to do.
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

# --- keep the raw token ids on every pipeline stage -------------------------
# The V4.1 MoE gate routes image sentinel tokens with `bias_vl` and hash
# experts with a token table, so it reads the raw token ids at every layer.
# The model declares that need as `requires_raw_input_tokens`. The model
# runner still clears `input_ids` on every stage after the first, and the gate
# then raises:
#
#   ValueError: DeepSeek V4 vision MoE routing requires input_ids.
#
# The ids are already built on every stage, so keep them where the model asks
# for them. The Engram hash on a later stage reads them too.
sub("vllm/v1/worker/gpu/cudagraph_utils.py",
    """            if not self.is_first_pp_rank:
                # Update for non-first PP ranks.
                model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None""",
    """            if not self.is_first_pp_rank:
                # Update for non-first PP ranks. Keep the raw token ids the
                # same way the model runner does.
                from vllm.model_executor.models.interfaces import (
                    requires_raw_input_tokens,
                )

                if not requires_raw_input_tokens(model):
                    model_inputs["input_ids"] = None
                model_inputs["inputs_embeds"] = None""")

sub("vllm/v1/worker/gpu/model_runner.py",
    """        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None""",
    """        if not self.is_first_pp_rank:
            # Update for non-first PP ranks.
            if not requires_raw_input_tokens(self.model):
                model_inputs["input_ids"] = None
            model_inputs["inputs_embeds"] = None""")

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
        )""",
    done_if='"kv_latent": torch.zeros(')

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
            )""",
    done_if='"kv_latent": self._relay_latent_buffer[')

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
                )""",
    done_if="for _relay in self.kv_group_relays:")

# 7. Let the shadow weights load under their source-layer checkpoint names.
sub(_M41,
    """        params_dict = dict(self.named_parameters())""",
    """        params_dict = dict(self.named_parameters())
        if self.kv_group_relays:
            from vllm.models.deepseek_v4_1.pp_kv_group_relay import (
                alias_source_params,
            )

            alias_source_params(list(self.kv_group_relays), params_dict)""",
    done_if="alias_source_params(list(self.kv_group_relays), params_dict)")
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

# --- carry the DSpark aux hidden states across the pipeline hop -------------
# The drafter reads the attention inputs of layers 37 to 39. The model
# collects them into a list and returns that list on the last stage only, so
# a stage that holds one of those layers drops it. The model runner refuses
# the whole configuration for that reason:
#
#   ValueError: DeepseekV41ForCausalLM does not support dspark with
#   pipeline parallelism
#
# `EagleModelMixin` already carries the pack and collect helpers. Wire them up
# the way the V4.0 model does, and declare the capability.
sub(_M41,
    """class DeepseekV4Model(nn.Module, EagleModelMixin):""",
    """class DeepseekV4Model(nn.Module, EagleModelMixin):
    supports_aux_hidden_states_over_pp = True
""")

sub(_M41,
    """        aux_hidden_states: list[torch.Tensor] = []""",
    """        remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors)
        aux_hidden_states: list[torch.Tensor] = []""")

sub(_M41,
    """                        if self.candidate_block_buffer is not None
                        else {}
                    ),
                }
            )""",
    """                        if self.candidate_block_buffer is not None
                        else {}
                    ),
                    **self.pack_local_aux_hidden_states(aux_hidden_states),
                }
            )""")

sub(_M41,
    """        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states""",
    """        aux_hidden_states = remote_aux + aux_hidden_states
        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states""")
print("dspark aux relay done")


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
sub("vllm/entrypoints/generate/base/protocol.py", _ALIAS_OLD, _ALIAS_NEW)
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
            # A V3.2 wrapper quoted inside the thinking block stays
            # thinking. Its own block keeps the reasoning content type, so
            # the text does not move into the answer.
            (ParserState.REASONING, "FOREIGN_START"): Transition(
                ParserState.FOREIGN_REASONING_BLOCK,
                (EventType.REASONING_CHUNK,),
            ),
            (
                ParserState.FOREIGN_REASONING_BLOCK,
                "FOREIGN_END",
            ): Transition(
                ParserState.REASONING,
                (EventType.REASONING_CHUNK,),
            ),
            (
                ParserState.FOREIGN_REASONING_BLOCK,
                "TOOL_START",
            ): Transition(
                ParserState.TOOL_PREAMBLE,
                (EventType.REASONING_END,),
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
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
""",
    """            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END_LENIENT"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),
""")

sub(_PARSER,
    """            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (),
            ),
""",
    """            (ParserState.TOOL_BETWEEN, "TOOL_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (),
            ),
            (ParserState.TOOL_BETWEEN, "TOOL_END_LENIENT"): Transition(
                ParserState.TOOL_BETWEEN,
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

# --- parser: the new state the reasoning-side foreign block needs -----------
sub("vllm/parser/engine/parser_engine_config.py",
    """    FOREIGN_BLOCK = auto()""",
    """    FOREIGN_BLOCK = auto()
    FOREIGN_REASONING_BLOCK = auto()""")

sub(_PARSER,
    """            ParserState.FOREIGN_BLOCK: EventType.TEXT_CHUNK,""",
    """            ParserState.FOREIGN_BLOCK: EventType.TEXT_CHUNK,
            ParserState.FOREIGN_REASONING_BLOCK: EventType.REASONING_CHUNK,""")

_SPE = "vllm/parser/engine/streaming_parser_engine.py"
sub(_SPE,
    """        elif self.state == ParserState.REASONING:
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT""",
    """        elif self.state in (
            ParserState.REASONING,
            ParserState.FOREIGN_REASONING_BLOCK,
        ):
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=self.tool_index)
            )
            self.state = ParserState.CONTENT
        elif self.state == ParserState.FOREIGN_BLOCK:
            self.state = ParserState.CONTENT""")

# --- parser: hold a recovered call until its invoke closes ------------------
# The engine commits a recovered call as soon as the name validates, so
# `<｜DSML｜ invoke name="bash">` written in prose commits on the closing
# quote of the tag. Keep the hold open through the arguments and commit only
# at the invoke close. Only a recovered call waits. A call that arrives in
# its proper envelope never enters the hold, so ordinary argument streaming
# is unchanged.
sub("vllm/parser/engine/parser_engine_config.py",
    """    validate_tool_name: bool = False""",
    """    validate_tool_name: bool = False
    # Commit a held tool call at this transition. A held call that leaves the
    # argument state through any other transition goes back to being text.
    commit_validated_tool_call: bool = False""")

sub(_PARSER,
    """            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
            ),""",
    """            (ParserState.TOOL_ARGS, "INVOKE_END"): Transition(
                ParserState.TOOL_BETWEEN,
                (EventType.TOOL_CALL_END,),
                commit_validated_tool_call=True,
            ),""")

# The name-completing transition no longer ends the hold. It validates the
# name, keeps the events buffered and moves on to the arguments.
sub(_SPE,
    """    def _resolve_hold(
        self,
        transition: Transition,
        value: str,
        token_count: int = 0,
    ) -> list[SemanticEvent]:
        \"\"\"End the hold window at the name-completing transition.\"\"\"
        name = "".join(self._held_name)
        allowed = self.allowed_tool_names
        if allowed is not None and name in allowed:
            events = self._held_events
            self._clear_hold()
            events.extend(self._run_transition(transition, value, token_count))
            return events
        return self._abort_hold("".join(self._held_raw) + value)""",
    """    def _resolve_hold(
        self,
        transition: Transition,
        value: str,
        token_count: int = 0,
    ) -> list[SemanticEvent]:
        \"\"\"Validate at the name, then commit at the invoke close.\"\"\"
        if self.state == ParserState.TOOL_NAME:
            name = "".join(self._held_name)
            allowed = self.allowed_tool_names
            if allowed is None or name not in allowed:
                return self._abort_hold("".join(self._held_raw) + value)
            self._held_raw.append(value)
            self._held_events.extend(
                self._run_transition(transition, value, token_count)
            )
            return []
        if self.state == ParserState.TOOL_ARGS:
            if not transition.commit_validated_tool_call:
                # Any other way out of the arguments means the invoke never
                # closed, so the whole run was text.
                return self._abort_hold("".join(self._held_raw) + value)
            self._held_raw.append(value)
            self._held_events.extend(
                self._run_transition(transition, value, token_count)
            )
            events = self._held_events
            self._clear_hold()
            return events
        return self._abort_hold("".join(self._held_raw) + value)""")

# Argument text, and the parameter closers that carry no transition, stay in
# the buffer while the hold is open.
sub(_SPE,
    """        if self.state == ParserState.MESSAGE_HEADER:
            self._message_header_buffer += text
            self._message_header_token_count += token_count
            return []
        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)""",
    """        if self._hold_active and self.state == ParserState.TOOL_ARGS:
            self._held_raw.append(text)
            self._held_events.append(
                SemanticEvent(
                    EventType.ARG_VALUE_CHUNK,
                    value=text,
                    tool_index=self.tool_index,
                    token_count=token_count,
                )
            )
            return []
        if self.state == ParserState.MESSAGE_HEADER:
            self._message_header_buffer += text
            self._message_header_token_count += token_count
            return []
        if self.state == ParserState.TOOL_ARGS:
            if self.config.tool_args_json:
                return self._feed_args_text(text)""")

# The end-of-stream abort already exists. It now also covers an invoke that
# never closed, so say that.
sub(_SPE,
    """            # Stream ended before the recovered tool name completed:
            # the held events never validated, so flush the raw text
            # as content in the pre-recovery state.""",
    """            # The stream ended before the recovered call completed, at
            # the name or at the invoke close. The held events never
            # committed, so flush the raw text as content in the
            # pre-recovery state.""")
print("validated tool-call hold done")

# --- parser: validate before the reasoning pass hands a call over ----------
# The reasoning pass runs the engine with `skip_tool_parsing` on. Every tool
# terminal then takes the blanket branch below, so a `<｜DSML｜ invoke name="`
# inside the thinking block ends the block and moves the rest of the turn into
# the answer, whatever the text turns out to be. Let a `validate_tool_name`
# transition run its hold instead. The hand-off to the tool pass then happens
# only for a complete call that names a declared tool and closes its invoke.
_SPE = "vllm/parser/engine/streaming_parser_engine.py"
sub(_SPE,
    """        if self.skip_tool_parsing and terminal in self._tool_terminals:""",
    """        if self.skip_tool_parsing and (
            transition.validate_tool_name or self._hold_active
        ):
            # A held call is still a candidate. Resolve it first, then let
            # the commit below decide what the reasoning pass emits.
            return self._apply_transition(transition, value, token_count)

        if self.skip_tool_parsing and terminal in self._tool_terminals:""")

# On commit inside the reasoning pass, emit the raw text rather than tool
# events. The tool pass parses that text again and builds the call from it.
sub(_SPE,
    """        if self.state == ParserState.TOOL_ARGS:
            if not transition.commit_validated_tool_call:
                # Any other way out of the arguments means the invoke never
                # closed, so the whole run was text.
                return self._abort_hold("".join(self._held_raw) + value)
            self._held_raw.append(value)""",
    """        if self.state == ParserState.TOOL_ARGS:
            if not transition.commit_validated_tool_call:
                # Any other way out of the arguments means the invoke never
                # closed, so the whole run was text.
                return self._abort_hold("".join(self._held_raw) + value)
            if self.skip_tool_parsing:
                return self._hand_off_hold(value)
            self._held_raw.append(value)""")

sub(_SPE,
    """    def _abort_hold(self, raw: str) -> list[SemanticEvent]:""",
    """    def _hand_off_hold(self, value: str) -> list[SemanticEvent]:
        \"\"\"Give a validated call to the tool pass as plain text.

        The reasoning pass owns no tool output, so it reports the end of the
        thinking block and repeats the call verbatim. The tool pass reads that
        text from the content state and builds the call there.
        \"\"\"
        raw = "".join(self._held_raw) + value
        prior_state = self._held_prior_state
        prior_tool_index = self._held_prior_tool_index
        self.state = ParserState.CONTENT
        self.tool_index = prior_tool_index
        self._recovered_tool_call = False
        self._clear_hold()
        events: list[SemanticEvent] = []
        if prior_state == ParserState.REASONING:
            events.append(
                SemanticEvent(EventType.REASONING_END, tool_index=prior_tool_index)
            )
        events.append(
            SemanticEvent(
                EventType.TEXT_CHUNK, value=raw, tool_index=prior_tool_index
            )
        )
        return events

    def _abort_hold(self, raw: str) -> list[SemanticEvent]:""")

# --- parser: the reasoning pass needs the declared tools -------------------
# `_check_skip_tool_parsing` records the declared names and whether the
# request forbids tool calls. Only the tool-side entry points call it, so the
# reasoning pass would validate a held call against a stale tool set. The
# request adjustment runs once per request, before any token, so do it there.
sub("vllm/parser/engine/adapters.py",
    """    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        return self._parser_engine.adjust_request(request)

    def has_engine_confirmed_reasoning_end(self) -> bool:""",
    """    def adjust_request(
        self,
        request: ChatCompletionRequest | ResponsesRequest,
    ) -> ChatCompletionRequest | ResponsesRequest:
        request = self._parser_engine.adjust_request(request)
        # The reasoning pass validates a recovered call, so it needs the
        # declared tool names and the suppression state too.
        self._parser_engine._check_skip_tool_parsing(request)
        return request

    def has_engine_confirmed_reasoning_end(self) -> bool:""")

# --- parser: flush a deferred reasoning tail at the end of the stream ------
# `finish_streaming` tests the deferred content only, so a trailing piece of
# reasoning held back for whitespace trimming never reaches the client.
sub("vllm/parser/engine/parser_engine.py",
    """        if events or self._deferred_content:
            return self._events_to_delta(events, finished=True)""",
    """        if events or self._deferred_content or self._deferred_reasoning:
            return self._strip_trailing_reasoning(
                self._events_to_delta(events, finished=True)
            )""")
print("reasoning-pass validation done")

print("parser reasoning-state recovery and lenient envelope done")


# --- sampler: hold the thinking block open for a minimum token count -------
# The V4.1 generation prompt ends with the `<think>` token, so every turn
# starts inside the thinking block. The checkpoint can close that block with
# its first generated token. The reasoning block is then empty, the model
# writes its deliberation into `content`, and the real `</think>` at the end
# reaches the parser in its content state, which absorbs it without an event.
#
# The thinking-token budget already finds the last `<think>` and counts the
# tokens after it, for the opposite purpose. These edits add a floor to the
# same kernel. Above the floor it forces the end marker as before. Below the
# floor it forbids the same token instead.
#
# This runs in the Model Runner V2 sampler. A custom logits processor cannot
# do the job, because `--logits-processors` forces Model Runner V1, and V1
# rejects DSpark speculative decoding.
_TB = "vllm/v1/worker/gpu/sample/thinking_budget.py"

sub(_TB,
    """from typing import TYPE_CHECKING

import numpy as np
""",
    """import os
from typing import TYPE_CHECKING

import numpy as np
""")

sub(_TB,
    """        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        if not self.enabled:
            return
""",
    """        self.enabled = bool(start_ids and end_ids and natural_end_ids)
        # The floor, in tokens. Zero keeps the stock behavior.
        self.min_thinking_tokens = int(
            os.environ.get("VLLM_MIN_THINKING_TOKENS", "0") or 0
        )
        if not self.enabled:
            return
""")

# Every request takes part once a floor is set, not only a request that asks
# for a budget.
sub(_TB,
    """        budget = sampling_params.thinking_token_budget
        self.use_thinking_budget[req_idx] = budget is not None
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
            self._reset_reqs.append(req_idx)
""",
    """        budget = sampling_params.thinking_token_budget
        self.use_thinking_budget[req_idx] = (
            budget is not None or self.min_thinking_tokens > 0
        )
        if budget is None:
            budget = -1
        else:
            budget = min(budget, _INT32_MAX)
        if self.use_thinking_budget[req_idx]:
            self._reset_reqs.append(req_idx)
""")

sub(_TB,
    """            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
        )


@triton.jit
def _load_effective_token(
""",
    """            self.natural_reasoning_end_token_ids,
            self.reasoning_end_token_ids,
            self.min_thinking_tokens,
        )


@triton.jit
def _load_effective_token(
""")

sub(_TB,
    """    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return
""",
    """    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    MAX_LEN: tl.constexpr,
    BLOCK: tl.constexpr,
    MIN_THINK: tl.constexpr,
):
    req_state_idx = tl.load(req_ids_ptr + tl.program_id(0))
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0 and MIN_THINK <= 0:
        return
""")

sub(_TB,
    """    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0:
        return
""",
    """    START_LEN: tl.constexpr,
    NATURAL_END_LEN: tl.constexpr,
    END_LEN: tl.constexpr,
    MIN_THINK: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(thinking_token_budget_ptr + req_state_idx)
    if budget < 0 and MIN_THINK <= 0:
        return
""")

sub(_TB,
    """    num_reasoning_tokens = effective_len - reasoning_start
    if num_reasoning_tokens < budget:
        return
""",
    """    num_reasoning_tokens = effective_len - reasoning_start
    below_floor = num_reasoning_tokens < MIN_THINK
    if not below_floor:
        if budget < 0:
            return
        if num_reasoning_tokens < budget:
            return
""")

sub(_TB,
    """    force_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    tl.store(logits_ptr + token_idx * logits_stride + force_token_id, 1.0e9)
""",
    """    marker_token_id = tl.load(reasoning_end_token_ids_ptr + end_prefix_len)
    # Above the floor this position carries the marker to force. Below the
    # floor it carries the token to forbid, so the block cannot close yet.
    fill = 1.0e9
    if below_floor:
        fill = -1.0e9
    tl.store(logits_ptr + token_idx * logits_stride + marker_token_id, fill)
""")

sub(_TB,
    """    reasoning_end_token_ids: torch.Tensor,
) -> None:
    num_tokens = logits.shape[0]
""",
    """    reasoning_end_token_ids: torch.Tensor,
    min_thinking_tokens: int = 0,
) -> None:
    num_tokens = logits.shape[0]
""")

sub(_TB,
    """        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
    )
""",
    """        MAX_LEN=max(start_len, natural_end_len),
        BLOCK=_COLD_SCAN_BLOCK,
        MIN_THINK=min_thinking_tokens,
    )
""")

sub(_TB,
    """        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
    )
""",
    """        START_LEN=start_len,
        NATURAL_END_LEN=natural_end_len,
        END_LEN=end_len,
        MIN_THINK=min_thinking_tokens,
    )
""")
print("thinking-token floor done")
