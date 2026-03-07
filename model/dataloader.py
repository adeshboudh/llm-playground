# model/dataloader.py

import os
import numpy as np
import torch
from pathlib import Path

class DataLoaderLite:
    """
    Reads binary uint16 shard files produced by ShardEncoder.
    Supports multi-process DDP training — each rank reads a different slice.

    Shard file format: flat array of uint16 token IDs, no header.
    Naming convention: shard_train_XXXX.bin / shard_val_XXXX.bin
    """
    def __init__(
        self,
        shards_dir: str,
        split: str,
        B: int,
        T: int,
        process_rank: int = 0,
        num_processes: int = 1,
    ):
        assert split in ("train", "val"), f"split must be 'train' or 'val', got {split!r}"
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split

        shards_path = Path(shards_dir)
        all_shards = sorted(shards_path.glob(f"shard_{split}_*.bin"))
        assert len(all_shards) > 0, (
            f"No shards found for split={split!r} in {shards_dir!r}. "
            f"Expected files matching shard_{split}_XXXX.bin"
        )
        self.shards = [str(s) for s in all_shards]
        self.reset()

    def reset(self) -> None:
        """Resets to the beginning of the first shard. Called at epoch start."""
        self.current_shard = 0
        self.tokens = self._load_shard(self.shards[self.current_shard])
        # Each rank starts at a different offset to avoid reading same tokens
        self.current_position = self.B * self.T * self.process_rank

    def _load_shard(self, path: str) -> torch.Tensor:
        """Loads a binary uint16 shard file into a long tensor."""
        tokens = np.fromfile(path, dtype=np.uint16).astype(np.int32)
        return torch.tensor(tokens, dtype=torch.long)
    
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (x, y) tensors of shape (B, T).
        x = input tokens, y = target tokens (shifted by 1).
        Automatically advances to the next shard when current is exhausted.
        """
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        # Advance position by all processes' worth of tokens
        self.current_position += B * T * self.num_processes

        # If next batch would go out of bounds, move to next shard
        if self.current_position + B * T * self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = self._load_shard(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank

        return x, y

    @property
    def num_shards(self) -> int:
        return len(self.shards)

    @property
    def total_tokens(self) -> int:
        """Approximate total tokens across all shards (based on file sizes)."""
        return sum(
            os.path.getsize(s) // 2  # uint16 = 2 bytes per token
            for s in self.shards
        )