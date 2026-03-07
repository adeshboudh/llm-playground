# tests/test_dataloader.py

import os
import struct
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path
from model.dataloader import DataLoaderLite


@pytest.fixture
def shard_dir(tmp_path):
    """Creates a temp dir with two train shards and one val shard."""
    # Each shard: 10,000 sequential uint16 tokens
    for i, split in enumerate([("train", 0), ("train", 1), ("val", 0)]):
        name, idx = split
        tokens = np.arange(idx * 10000, idx * 10000 + 10000, dtype=np.uint16)
        path = tmp_path / f"shard_{name}_{idx:04d}.bin"
        tokens.tofile(str(path))
    return str(tmp_path)


def test_loads_correct_split(shard_dir):
    loader = DataLoaderLite(shard_dir, split="train", B=4, T=16)
    assert loader.num_shards == 2


def test_val_split_loads(shard_dir):
    loader = DataLoaderLite(shard_dir, split="val", B=4, T=16)
    assert loader.num_shards == 1


def test_next_batch_shape(shard_dir):
    B, T = 4, 16
    loader = DataLoaderLite(shard_dir, split="train", B=B, T=T)
    x, y = loader.next_batch()
    assert x.shape == (B, T)
    assert y.shape == (B, T)


def test_targets_are_shifted_by_one(shard_dir):
    """y must equal x shifted left by 1 — the language modeling target."""
    loader = DataLoaderLite(shard_dir, split="train", B=2, T=8)
    x, y = loader.next_batch()
    assert torch.all(y[:, :-1] == x[:, 1:]), "y is not x shifted by 1"


def test_missing_shards_raises(tmp_path):
    """Valid split but no shard files present."""
    with pytest.raises(AssertionError, match="No shards found"):
        DataLoaderLite(str(tmp_path), split="train", B=4, T=16)


def test_reset_returns_to_start(shard_dir):
    loader = DataLoaderLite(shard_dir, split="train", B=4, T=16)
    x1, _ = loader.next_batch()
    loader.next_batch()
    loader.reset()
    x2, _ = loader.next_batch()
    assert torch.all(x1 == x2), "reset() did not return to start"


def test_multiprocess_ranks_differ(shard_dir):
    """Different ranks must read different token slices."""
    loader0 = DataLoaderLite(shard_dir, split="train", B=4, T=16,
                             process_rank=0, num_processes=2)
    loader1 = DataLoaderLite(shard_dir, split="train", B=4, T=16,
                             process_rank=1, num_processes=2)
    x0, _ = loader0.next_batch()
    x1, _ = loader1.next_batch()
    assert not torch.all(x0 == x1), "ranks 0 and 1 read identical tokens"


def test_shard_wraps_around(shard_dir):
    """Exhaust shard 0 and verify it wraps to shard 1 without error."""
    B, T = 4, 16
    loader = DataLoaderLite(shard_dir, split="train", B=B, T=T)
    tokens_in_shard = 10_000
    steps_to_exhaust = tokens_in_shard // (B * T) + 5  # go past the boundary
    for _ in range(steps_to_exhaust):
        x, y = loader.next_batch()
    assert x.shape == (B, T), "Batch shape broke after shard wrap"


def test_total_tokens(shard_dir):
    loader = DataLoaderLite(shard_dir, split="train", B=4, T=16)
    # 2 shards × 10,000 tokens each
    assert loader.total_tokens == 20_000
