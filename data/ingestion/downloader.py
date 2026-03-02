import logging

from dataclasses import dataclass
from typing import Iterator
from datasets import load_dataset, IterableDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

@dataclass
class DownloaderConfig:
    source: str           # HF dataset name
    subset: str           # config/subset name
    num_samples: int
    text_column: str
    streaming: bool = True

class DatasetDownloader:
    """
    Streams documents from HuggingFace datasets.
    Never loads entire dataset into RAM.
    """

    def __init__(self, config: DownloaderConfig):
        self.config = config
        self.validate_config()
        self._dataset: IterableDataset | None = None
        logger.info(f"Initialized DatasetDownloader for {config.source}/{config.subset}")

    def validate_config(self) -> None:
        """Validates the downloader configuration."""
        if self.config.num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {self.config.num_samples}")
        if not self.config.text_column:
            raise ValueError("text_column cannot be empty")


    def load(self) -> "DatasetDownloader":
        # TODO: load HF dataset, log how many samples will be pulled
        """Initializes streaming connection. Call before iterate()."""
        self._dataset = load_dataset(
            self.config.source,
            name=self.config.subset,
            split="train",
            streaming=self.config.streaming,
            trust_remote_code=True
        )
        logger.info(f"Dataset loaded. Will pull up to {self.config.num_samples} samples.")
        return self

    def iterate(self) -> Iterator[str]:
        # TODO: yield text strings with tqdm progress bar
        # skip empty strings silently, log total yielded at end
        """
        Yields raw text strings one document at a time.

        Yields:
            str: Raw document text

        Raises:
            RuntimeError: If load() was not called first
        """
        if self._dataset is None:
            raise RuntimeError("Call load() before iterate()")
        count = 0
        for doc in self._dataset:
            if count >= self.config.num_samples:
                break
            text = doc.get(self.config.text_column, "").strip()
            if text:
                yield text
                count += 1
                pbar.update(1)

    def __repr__(self) -> str:
        return (f"DatasetDownloader(source='{self.config.source}', "
                f"subset='{self.config.subset}', num_samples={self.config.num_samples})")

    def __len__(self) -> int:
        return self.config.num_samples