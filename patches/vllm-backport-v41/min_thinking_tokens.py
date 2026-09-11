# SPDX-License-Identifier: Apache-2.0
"""Hold the thinking block open for a minimum number of tokens.

DeepSeek-V4.1-Flash renders a generation prompt that ends with the `<think>`
token, so every turn starts inside the thinking block. At long context the
checkpoint can sample `</think>` as its very first token. The reasoning block
is then empty, the model writes its deliberation into the content channel, and
the real `</think>` at the end of that deliberation reaches the parser in its
content state, where it is absorbed without an event. The client shows the
thinking as chat text, and it feeds that text back on the next turn.

This processor masks the `</think>` token until the request has generated
`VLLM_MIN_THINKING_TOKENS` tokens. It only tracks a request whose prompt ends
inside a thinking block.

It subclasses `MinTokensLogitsProcessor` on purpose. The rejection sampler
applies only that class under speculative decoding, so a plain
`LogitsProcessor` would be skipped while DSpark is on. The base class supplies
the batch bookkeeping, the mask, and the draft-row expansion. This subclass
replaces the per-request decision only.

Load it with:
  --logits-processors vllm_min_thinking:MinThinkingTokensLogitsProcessor
"""

import os
from collections.abc import Sequence

from vllm import SamplingParams
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor

DEFAULT_THINK_START_TOKEN_ID = 128821
DEFAULT_THINK_END_TOKEN_ID = 128822


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


class MinThinkingTokensLogitsProcessor(MinTokensLogitsProcessor):
    """Ban the end-of-thinking token for the first N tokens of a turn."""

    def __init__(self, vllm_config, device, is_pin_memory) -> None:
        super().__init__(vllm_config, device, is_pin_memory)
        self.min_thinking_tokens = _env_int("VLLM_MIN_THINKING_TOKENS", 0)

        start_ids, end_ids = self._marker_ids(vllm_config)
        self.think_start_token_ids = start_ids
        self.think_end_token_ids = end_ids

    @staticmethod
    def _marker_ids(vllm_config) -> tuple[list[int], list[int]]:
        """Read the thinking markers from the reasoning configuration.

        The environment wins, so the operator can correct a bad reading
        without a rebuild.
        """
        config = getattr(vllm_config, "reasoning_config", None)
        start = list(getattr(config, "reasoning_start_token_ids", None) or [])
        end = list(getattr(config, "reasoning_end_token_ids", None) or [])
        if not start:
            start = [DEFAULT_THINK_START_TOKEN_ID]
        if not end:
            end = [DEFAULT_THINK_END_TOKEN_ID]
        start_env = os.environ.get("VLLM_THINK_START_TOKEN_ID")
        end_env = os.environ.get("VLLM_THINK_END_TOKEN_ID")
        if start_env:
            start = [int(start_env)]
        if end_env:
            end = [int(end_env)]
        return start, end

    @staticmethod
    def _last_index(haystack: Sequence[int], needle: Sequence[int]) -> int:
        if not needle:
            return -1
        for i in range(len(haystack) - len(needle), -1, -1):
            if list(haystack[i : i + len(needle)]) == list(needle):
                return i
        return -1

    def _prompt_is_inside_think(self, prompt_tok_ids: Sequence[int] | None) -> bool:
        if not prompt_tok_ids:
            return False
        last_start = self._last_index(prompt_tok_ids, self.think_start_token_ids)
        if last_start < 0:
            return False
        last_end = self._last_index(prompt_tok_ids, self.think_end_token_ids)
        return last_start > last_end

    # The base class calls this through `process_dict_updates`. Returning
    # `None` leaves the request untracked.
    def add_request(  # type: ignore[override]
        self,
        params: SamplingParams,
        prompt_tok_ids: list[int] | None,
        output_tok_ids: list[int],
    ) -> tuple[int, Sequence[int], set[int]] | None:
        minimum = self.min_thinking_tokens
        if minimum <= 0 or len(output_tok_ids) >= minimum:
            return None
        if not self._prompt_is_inside_think(prompt_tok_ids):
            return None
        return minimum, output_tok_ids, set(self.think_end_token_ids)
