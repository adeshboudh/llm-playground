# tests/test_encoder.py

import sys
import numpy as np
import pytest
import shutil
import uuid
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer.encoder import ShardEncoder


class MockTokenizer:
    def __init__(self):
        self.vocab = {
            "hello": [10, 11],
            "world": [12],
            "<|endoftext|>": [50256]
        }

    def encode(self, text: str) -> list[int]:
        return list(self.vocab.get(text, [99]))


@pytest.fixture
def temp_output_dir():
    d = Path("scratch") / f"test_encoder_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=False)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------
# 1. Basic encoding test
# ---------------------------------------------------------
def test_shard_encoder_basic(temp_output_dir):
    tokenizer = MockTokenizer()

    encoder = ShardEncoder(
        tokenizer=tokenizer,
        output_dir=str(temp_output_dir),
        shard_size=4,
        split="test",
    )

    texts = ["hello", "world", "hello"]

    stats = encoder.encode_stream(iter(texts))
    assert stats["total_tokens"] == 8
    assert stats["num_shards"] == 2

    shard_files = list(temp_output_dir.glob("*.bin"))
    assert len(shard_files) == 2

    data0 = np.fromfile(shard_files[0], dtype=np.uint16)
    data1 = np.fromfile(shard_files[1], dtype=np.uint16)

    assert np.array_equal(data0, np.array([10, 11, 50256, 12], dtype=np.uint16))
    assert np.array_equal(data1, np.array([50256, 10, 11, 50256], dtype=np.uint16))


# ---------------------------------------------------------
# 2. Multi shard test
# ---------------------------------------------------------
def test_shard_encoder_multi_shard(temp_output_dir):
    tokenizer = MockTokenizer()

    encoder = ShardEncoder(
        tokenizer=tokenizer,
        output_dir=str(temp_output_dir),
        shard_size=4,
        split="train",
    )

    texts = ["hello", "world", "hello"]

    stats = encoder.encode_stream(iter(texts))

    assert stats["total_tokens"] == 8
    assert stats["num_shards"] == 2

    shard_files = sorted(temp_output_dir.glob("*.bin"))
    assert len(shard_files) == 2

    data0 = np.fromfile(shard_files[0], dtype=np.uint16)
    data1 = np.fromfile(shard_files[1], dtype=np.uint16)

    assert np.array_equal(data0, np.array([10, 11, 50256, 12], dtype=np.uint16))
    assert np.array_equal(data1, np.array([50256, 10, 11, 50256], dtype=np.uint16))


# ---------------------------------------------------------
# 3. Partial final shard
# ---------------------------------------------------------
def test_shard_encoder_partial_final_shard(temp_output_dir):
    tokenizer = MockTokenizer()

    encoder = ShardEncoder(
        tokenizer=tokenizer,
        output_dir=str(temp_output_dir),
        shard_size=100,
        split="val",
    )

    texts = ["world"]

    stats = encoder.encode_stream(iter(texts))

    assert stats["total_tokens"] == 2
    assert stats["num_shards"] == 1

    path = Path(stats["output_dir"]) / "shard_val_0000.bin"

    assert path.exists()

    data = np.fromfile(path, dtype=np.uint16)

    assert len(data) == 2
    assert data[1] == 50256


# ---------------------------------------------------------
# 4. Check shard dtype is uint16
# ---------------------------------------------------------
def test_shard_dtype_is_uint16(temp_output_dir):
    tokenizer = MockTokenizer()

    encoder = ShardEncoder(
        tokenizer=tokenizer,
        output_dir=str(temp_output_dir),
        shard_size=10,
    )

    encoder.encode_stream(iter(["hello"]))

    shard_file = next(temp_output_dir.glob("*.bin"))

    data = np.fromfile(shard_file, dtype=np.uint16)

    assert data.dtype == np.uint16


# ---------------------------------------------------------
# 5. Verify correct shard count
# ---------------------------------------------------------
def test_shard_count_exact(temp_output_dir):
    tokenizer = MockTokenizer()

    encoder = ShardEncoder(
        tokenizer=tokenizer,
        output_dir=str(temp_output_dir),
        shard_size=3,
    )

    texts = ["hello", "world"]

    stats = encoder.encode_stream(iter(texts))

    # hello -> 3 tokens
    # world -> 2 tokens
    # total = 5 → shards: [3,2]

    assert stats["num_shards"] == 2

    shard_files = list(temp_output_dir.glob("*.bin"))

    assert len(shard_files) == 2


# ---------------------------------------------------------
# 6. Ensure EOT token appears between documents
# ---------------------------------------------------------
def test_eot_between_documents(temp_output_dir):
    tokenizer = MockTokenizer()

    encoder = ShardEncoder(
        tokenizer=tokenizer,
        output_dir=str(temp_output_dir),
        shard_size=20,
    )

    texts = ["hello", "world"]

    encoder.encode_stream(iter(texts))

    shard_file = next(temp_output_dir.glob("*.bin"))

    data = np.fromfile(shard_file, dtype=np.uint16)

    # Expected sequence:
    # hello -> [10,11] + EOT
    # world -> [12] + EOT

    expected = np.array(
        [10, 11, 50256, 12, 50256],
        dtype=np.uint16
    )

    assert np.array_equal(data, expected)