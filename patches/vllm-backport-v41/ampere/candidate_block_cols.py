"""Turn candidate blocks into the paged-attention block columns to visit.

The V4.1 indexer at `candidate_source_layer_id` publishes its top blocks of
compressed positions. Every later indexer masks its scores to those blocks,
so a score outside them cannot survive. Computing those scores is waste.

This module maps the candidate blocks of one step to the key-value block
columns that hold them. The logits kernel then visits only those columns.
The mask that follows writes minus infinity everywhere else, so a column
this map leaves out holds the same value it would have held before.

A candidate block covers `candidate_block_size` compressed positions, which
is far fewer than the `block_size` positions of a key-value block. Several
candidate blocks therefore fall in one column, so the map must drop the
duplicates or the kernel recomputes the same column many times.
"""

import torch

# The four consumer layers of one step read the same candidate blocks, so
# the map is built once and reused. The key is the buffer itself, the row
# count and the block geometry. A step that writes new candidates writes new
# values into the same buffer, so the key also carries a counter that the
# writer bumps.
_CACHE: dict = {}
_EPOCH = [0]


def new_step() -> None:
    """Drop the cached map. Call this when the candidate buffer changes."""
    _EPOCH[0] += 1


def block_columns(
    candidate_blocks: torch.Tensor,
    candidate_block_size: int,
    block_size: int,
    num_columns: int,
) -> torch.Tensor:
    """The key-value block columns each row needs, padded with -1.

    Args:
        candidate_blocks: [rows, k] int32. A negative entry is empty.
        candidate_block_size: compressed positions per candidate block.
        block_size: compressed positions per key-value block.
        num_columns: block columns the block table holds.

    Returns:
        [rows, max_selected] int32. Each row lists its columns, ascending,
        and pads the tail with -1.
    """
    key = (
        candidate_blocks.data_ptr(),
        candidate_blocks.shape,
        num_columns,
        _EPOCH[0],
    )
    hit = _CACHE.get(key)
    if hit is not None:
        return hit

    rows = candidate_blocks.shape[0]
    per_column = max(1, block_size // candidate_block_size)
    columns = candidate_blocks.to(torch.int64) // per_column
    keep = (candidate_blocks >= 0) & (columns < num_columns)

    # One flag per column, then a running sum to place each survivor. An
    # empty candidate carries a negative value, so send it to a bin past the
    # last column. A clamp would send it to column 0 and could clear a flag
    # a real candidate set there.
    flags = torch.zeros(
        (rows, num_columns + 1), dtype=torch.bool, device=candidate_blocks.device
    )
    columns = torch.where(keep, columns.clamp(0, num_columns - 1), num_columns)
    flags.scatter_(1, columns, torch.ones_like(columns, dtype=torch.bool))
    flags = flags[:, :num_columns]
    counts = flags.sum(1)
    width = int(counts.max().item())
    if width == 0:
        width = 1
    slot = flags.cumsum(1) - 1

    grid = torch.arange(
        num_columns, device=flags.device, dtype=torch.int32
    ).expand(rows, num_columns)
    # Column `width` is a bin for the columns this row does not need. Every
    # position writes somewhere, so the unwanted ones write into the bin and
    # the slice below drops it.
    out = torch.full(
        (rows, width + 1), -1, dtype=torch.int32, device=flags.device
    )
    slot = torch.where(flags, slot, torch.full_like(slot, width))
    out.scatter_(1, slot, grid)
    out = out[:, :width].contiguous()

    _CACHE.clear()
    _CACHE[key] = out
    return out
