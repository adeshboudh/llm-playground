# tests/test_tokenizer_trainer.py
import sys
import uuid
from pathlib import Path

import pytest

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.tokenizer.trainer import BPETokenizerTrainer


def small_corpus():
    return [
        "hello world",
        "this is a tiny corpus",
        "tokenizers should train quickly",
    ]


def test_train_on_small_corpus_produces_non_empty_vocab():
    trainer = BPETokenizerTrainer(vocab_size=100, min_frequency=1)
    trainer.train(iter(small_corpus()))

    assert trainer.vocab_size > 0


def test_encode_decode_roundtrip_returns_original_text():
    trainer = BPETokenizerTrainer(vocab_size=200, min_frequency=1)
    trainer.train(iter(small_corpus()))

    text = "hello world"
    encoded = trainer.encode(text)
    decoded = trainer.decode(encoded)

    assert decoded == text


def _workspace_temp_file() -> Path:
    temp_dir = Path("tests") / ".tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"tokenizer_{uuid.uuid4().hex}.json"


def test_save_creates_file_on_disk():
    trainer = BPETokenizerTrainer(vocab_size=100, min_frequency=1)
    trainer.train(iter(small_corpus()))

    output_path = _workspace_temp_file()
    trainer.save(str(output_path))

    assert output_path.exists()
    assert output_path.is_file()
    output_path.unlink()


def test_load_restores_working_tokenizer():
    trainer = BPETokenizerTrainer(vocab_size=200, min_frequency=1)
    trainer.train(iter(small_corpus()))

    output_path = _workspace_temp_file()
    trainer.save(str(output_path))

    loaded = BPETokenizerTrainer.load(str(output_path))
    ids = loaded.encode("hello world")

    assert isinstance(ids, list)
    assert loaded.decode(ids) == "hello world"
    output_path.unlink()


def test_encode_before_train_raises_runtime_error():
    trainer = BPETokenizerTrainer()

    with pytest.raises(RuntimeError):
        trainer.encode("hello world")


def test_decode_before_train_raises_runtime_error():
    trainer = BPETokenizerTrainer()

    with pytest.raises(RuntimeError):
        trainer.decode([1, 2, 3])


def test_special_tokens_present_after_training():
    special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
    trainer = BPETokenizerTrainer(
        vocab_size=100,
        min_frequency=1,
        special_tokens=special_tokens,
    )
    trainer.train(iter(small_corpus()))

    vocab = trainer.get_vocab()
    for token in special_tokens:
        assert token in vocab
