# data/tokenizer/trainer.py

from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


class BPETokenizerTrainer:
    """
    Trains a GPT-2 style Byte-Level BPE tokenizer from scratch.
    Accepts a text iterator and does not write intermediate files.
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        special_tokens: list[str] | None = None,
    ):
        self.target_vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or [
            "<|endoftext|>",
            "<|pad|>",
            "<|unk|>",
            "<|im_start|>",
            "<|im_end|>",
        ]
        self.tokenizer: Tokenizer | None = None

    def _is_trained(self) -> bool:
        """Returns True if the tokenizer has been trained or loaded."""
        return self.tokenizer is not None

    def _require_trained(self) -> None:
        """Raises RuntimeError if the tokenizer is not trained."""
        if not self._is_trained():
            raise RuntimeError("Tokenizer is not trained. Call train() or load() first.")

    def train(self, text_iterator: Iterator[str]) -> "BPETokenizerTrainer":
        """Trains the tokenizer from an iterator of text samples."""
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.normalizer = NFC()
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.target_vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=False,
        )
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        self.tokenizer = tokenizer
        return self

    def save(self, path: str) -> None:
        """Saves tokenizer.json to disk, creating parent directories if needed."""
        self._require_trained()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizerTrainer":
        """Loads a tokenizer from a file path."""
        instance = cls()
        instance.tokenizer = Tokenizer.from_file(path)
        return instance

    def encode(self, text: str) -> list[int]:
        """Encodes text into a list of token IDs."""
        self._require_trained()
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """Decodes a list of token IDs into text."""
        self._require_trained()
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        """Returns the size of the tokenizer's vocabulary."""
        self._require_trained()
        return len(self.tokenizer.get_vocab())

    def get_vocab(self) -> dict[str, int]:
        """Returns the tokenizer's vocabulary as a dictionary."""
        self._require_trained()
        return self.tokenizer.get_vocab()
