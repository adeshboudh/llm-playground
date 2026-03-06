# data/tokenizer/encoder.py

import numpy as np
from pathlib import Path
from typing import Iterator


class ShardEncoder:
    """
    Tokenizes a text stream and writes binary .bin shards.
    Each shard = exactly shard_size_tokens uint16 token IDs.
    Compatible with nanoGPT's data loader format.
    """

    def __init__(
        self,
        tokenizer,
        output_dir: str,
        shard_size: int = 100_000_000,
        split: str = "train",
        dtype=np.uint16,
    ):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.shard_size = shard_size
        self.split = split
        self.dtype = dtype

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # preallocated buffer
        self._buffer = np.empty((self.shard_size,), dtype=self.dtype)

        self._shard_idx = 0
        self._token_count = 0
        self._total_tokens = 0

    def _flush_shard(self) -> None:
        """Write current buffer to disk."""
        path = self.output_dir / f"shard_{self.split}_{self._shard_idx:04d}.bin"

        self._buffer[:self._token_count].tofile(path)

        print(f"Wrote {path.name} ({self._token_count:,} tokens)")

        self._shard_idx += 1
        self._token_count = 0

    def encode_stream(self, text_iterator: Iterator[str]) -> dict:
        """
        Encodes text stream and writes shards.
        """

        # Correct EOT handling
        EOT_ID = self.tokenizer.encode("<|endoftext|>")[0]
        for text in text_iterator:
            print("lno: 58", text)
            ids = list(self.tokenizer.encode(text))
            ids.append(EOT_ID)

            pos = 0
            n = len(ids)

            while pos < n:

                space_left = self.shard_size - self._token_count
                take = min(space_left, n - pos)

                # copy tokens into numpy buffer
                self._buffer[self._token_count:self._token_count + take] = ids[pos:pos + take]

                self._token_count += take
                pos += take
                self._total_tokens += take

                if self._token_count == self.shard_size:
                    self._flush_shard()

        # final partial shard
        if self._token_count > 0:
            self._flush_shard()

        return {
            "total_tokens": self._total_tokens,
            "num_shards": self._shard_idx,
            "output_dir": str(self.output_dir),
        }