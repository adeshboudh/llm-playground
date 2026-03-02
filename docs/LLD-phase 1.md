# LLD: Phase 1 — Data Layer

**Version:** 1.0 | **Phase:** Pre-Training Data Pipeline | **Compute:** Colab Free CPU

---

## 1. Module Map

```
data/
├── ingestion/
│   ├── __init__.py
│   ├── downloader.py        # DatasetDownloader
│   └── sampler.py           # CorpusSampler
├── cleaning/
│   ├── __init__.py
│   ├── filters.py           # FilterPipeline + all Filter classes
│   ├── deduplicator.py      # MinHashDeduplicator
│   └── perplexity_filter.py # KenLMFilter
├── tokenizer/
│   ├── __init__.py
│   ├── trainer.py           # BPETokenizerTrainer
│   ├── encoder.py           # ShardEncoder
│   └── validator.py         # TokenizerValidator
├── registry/
│   ├── __init__.py
│   └── hub_pusher.py        # HFHubPusher
├── configs/
│   └── data_config.yaml     # All hyperparams in one place
├── tests/
│   ├── test_filters.py
│   ├── test_tokenizer.py
│   └── test_dedup.py
└── pipeline.py              # Orchestrator — runs all steps in order
```

---

## 2. Configuration Contract

All hyperparameters live in one YAML — no magic numbers in code.

```yaml
# configs/data_config.yaml

ingestion:
  source: "HuggingFaceFW/fineweb-edu" # HF dataset name
  subset: "sample-10BT" # Start with 10B token sample
  num_samples: 500_000 # Documents to pull
  text_column: "text"
  streaming: true # Don't download entire dataset

cleaning:
  min_length: 100 # chars
  max_length: 100_000 # chars
  min_avg_word_length: 3
  max_avg_word_length: 10
  max_symbol_to_word_ratio: 0.1
  max_bullet_lines_ratio: 0.9
  max_ellipsis_lines_ratio: 0.3
  dedup_enabled: true
  minhash_num_perm: 128
  minhash_threshold: 0.85 # Jaccard similarity threshold
  kenlm_enabled: false # Skip if kenlm not installed
  kenlm_max_perplexity: 1500

tokenizer:
  vocab_size: 32_000
  min_frequency: 2
  special_tokens:
    - "<|endoftext|>"
    - "<|pad|>"
    - "<|unk|>"
    - "<|im_start|>"
    - "<|im_end|>"
  shard_size_tokens: 100_000_000 # 100M tokens per shard

registry:
  hf_repo_id: "{username}/llm-playground-data-v1"
  push_tokenizer: true
  push_shards: true
```

---

## 3. Class Interfaces

### 3.1 `DatasetDownloader`

```python
# data/ingestion/downloader.py

from dataclasses import dataclass
from typing import Iterator
from datasets import load_dataset, IterableDataset

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
        self._dataset: IterableDataset | None = None

    def load(self) -> "DatasetDownloader":
        """Initializes streaming connection. Call before iterate()."""
        self._dataset = load_dataset(
            self.config.source,
            name=self.config.subset,
            split="train",
            streaming=self.config.streaming,
            trust_remote_code=True
        )
        return self

    def iterate(self) -> Iterator[str]:
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

    def __len__(self) -> int:
        return self.config.num_samples
```

---

### 3.2 `FilterPipeline` + Individual Filters

```python
# data/cleaning/filters.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
import re
import unicodedata

# ── Base Filter Interface ──────────────────────────────────────

class BaseFilter(ABC):
    """All filters must implement this interface."""

    @abstractmethod
    def is_valid(self, text: str) -> bool:
        """Returns True if text passes the filter."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

# ── Concrete Filters ───────────────────────────────────────────

@dataclass
class LengthFilter(BaseFilter):
    min_length: int = 100
    max_length: int = 100_000

    @property
    def name(self): return "length_filter"

    def is_valid(self, text: str) -> bool:
        return self.min_length <= len(text) <= self.max_length


@dataclass
class WordLengthFilter(BaseFilter):
    min_avg: float = 3.0
    max_avg: float = 10.0

    @property
    def name(self): return "word_length_filter"

    def is_valid(self, text: str) -> bool:
        words = text.split()
        if not words:
            return False
        avg = sum(len(w) for w in words) / len(words)
        return self.min_avg <= avg <= self.max_avg


@dataclass
class SymbolRatioFilter(BaseFilter):
    max_ratio: float = 0.1   # symbols / words

    @property
    def name(self): return "symbol_ratio_filter"

    def is_valid(self, text: str) -> bool:
        words = text.split()
        if not words:
            return False
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return (symbols / len(words)) <= self.max_ratio


@dataclass
class BulletLinesFilter(BaseFilter):
    max_ratio: float = 0.9

    @property
    def name(self): return "bullet_lines_filter"

    def is_valid(self, text: str) -> bool:
        lines = text.splitlines()
        if not lines:
            return False
        bullet_lines = sum(1 for l in lines if l.strip().startswith(("•", "-", "*", "·")))
        return (bullet_lines / len(lines)) <= self.max_ratio


@dataclass
class AlphanumericFilter(BaseFilter):
    min_ratio: float = 0.7   # alphanumeric chars / total chars

    @property
    def name(self): return "alphanum_filter"

    def is_valid(self, text: str) -> bool:
        if not text:
            return False
        alphanum = sum(1 for c in text if c.isalnum())
        return (alphanum / len(text)) >= self.min_ratio


# ── Pipeline Orchestrator ──────────────────────────────────────

class FilterPipeline:
    """
    Runs a sequence of filters. Short-circuits on first failure.
    Tracks rejection stats per filter for observability.
    """

    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters
        self.stats: dict[str, int] = {f.name: 0 for f in filters}
        self.total_seen = 0
        self.total_passed = 0

    def apply(self, text: str) -> bool:
        """
        Returns True if text passes ALL filters.
        Increments rejection counter for failing filter.
        """
        self.total_seen += 1
        for f in self.filters:
            if not f.is_valid(text):
                self.stats[f.name] += 1
                return False
        self.total_passed += 1
        return True

    def rejection_report(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "total_passed": self.total_passed,
            "pass_rate": round(self.total_passed / max(self.total_seen, 1), 4),
            "rejections_by_filter": self.stats
        }
```

---

### 3.3 `MinHashDeduplicator`

```python
# data/cleaning/deduplicator.py

from datasketch import MinHash, MinHashLSH
import re

class MinHashDeduplicator:
    """
    Exact-near-duplicate detection using MinHash + LSH.
    Operates in streaming fashion — no need to hold all docs in RAM.

    Jaccard similarity threshold: docs above this are duplicates.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85):
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._seen_count = 0
        self._dup_count = 0

    def _shingle(self, text: str, k: int = 5) -> set[str]:
        """Generates k-character shingles from text."""
        text = re.sub(r'\s+', ' ', text.lower())
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def _make_minhash(self, text: str) -> MinHash:
        m = MinHash(num_perm=self.num_perm)
        for shingle in self._shingle(text):
            m.update(shingle.encode('utf-8'))
        return m

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """
        Returns True if text is a near-duplicate of a previously seen doc.
        If not duplicate, registers this doc in the LSH index.

        Args:
            text: Document text
            doc_id: Unique identifier (e.g., index or URL hash)
        """
        self._seen_count += 1
        mh = self._make_minhash(text)
        results = self.lsh.query(mh)
        if results:
            self._dup_count += 1
            return True
        self.lsh.insert(doc_id, mh)
        return False

    @property
    def dedup_rate(self) -> float:
        return self._dup_count / max(self._seen_count, 1)
```

---

### 3.4 `BPETokenizerTrainer`

```python
# data/tokenizer/trainer.py

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.normalizers import NFC
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from typing import Iterator

class BPETokenizerTrainer:
    """
    Trains a GPT-2 style Byte-Level BPE tokenizer from scratch.
    Accepts a text iterator — never writes intermediate files.
    """

    def __init__(
        self,
        vocab_size: int = 32_000,
        min_frequency: int = 2,
        special_tokens: list[str] = None
    ):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens or [
            "<|endoftext|>", "<|pad|>", "<|unk|>",
            "<|im_start|>", "<|im_end|>"
        ]
        self.tokenizer: Tokenizer | None = None

    def train(self, text_iterator: Iterator[str]) -> "BPETokenizerTrainer":
        """
        Trains BPE tokenizer on a stream of text.

        Args:
            text_iterator: Any iterator yielding raw text strings

        Returns:
            self (for chaining)
        """
        tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
        tokenizer.normalizer = NFC()                    # Unicode normalization
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens,
            show_progress=True
        )
        tokenizer.train_from_iterator(text_iterator, trainer=trainer)
        self.tokenizer = tokenizer
        return self

    def save(self, path: str) -> None:
        """Saves tokenizer.json to disk."""
        assert self.tokenizer, "Train first."
        self.tokenizer.save(path)

    def encode(self, text: str) -> list[int]:
        assert self.tokenizer, "Train first."
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        assert self.tokenizer, "Train first."
        return self.tokenizer.decode(ids)
```

---

### 3.5 `ShardEncoder`

```python
# data/tokenizer/encoder.py

import numpy as np
from pathlib import Path
from typing import Iterator

class ShardEncoder:
    """
    Tokenizes a text stream and writes binary .bin shards.
    Each shard = exactly shard_size_tokens uint16 token IDs.
    Compatible with nanoGPT's data loader format.

    File naming: shard_train_0000.bin, shard_train_0001.bin, ...
    """

    def __init__(
        self,
        tokenizer,              # Any object with .encode(text) -> list[int]
        output_dir: str,
        shard_size: int = 100_000_000,   # 100M tokens per shard
        split: str = "train",
        dtype = np.uint16
    ):
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.split = split
        self.dtype = dtype

        self._buffer: list[int] = []
        self._shard_idx = 0
        self._total_tokens = 0

    def _flush_shard(self) -> None:
        """Writes current buffer as a .bin shard file."""
        tokens = np.array(self._buffer[:self.shard_size], dtype=self.dtype)
        path = self.output_dir / f"shard_{self.split}_{self._shard_idx:04d}.bin"
        tokens.tofile(str(path))
        self._buffer = self._buffer[self.shard_size:]
        self._shard_idx += 1
        print(f"  Wrote {path.name} ({len(tokens):,} tokens)")

    def encode_stream(self, text_iterator: Iterator[str]) -> dict:
        """
        Encodes all texts and writes shards.

        Returns:
            dict with total_tokens, num_shards, output_dir
        """
        EOT_ID = self.tokenizer.encode("<|endoftext|>")

        for text in text_iterator:
            ids = self.tokenizer.encode(text) + [EOT_ID]
            self._buffer.extend(ids)
            self._total_tokens += len(ids)

            while len(self._buffer) >= self.shard_size:
                self._flush_shard()

        # Write remaining tokens as final partial shard
        if self._buffer:
            tokens = np.array(self._buffer, dtype=self.dtype)
            path = self.output_dir / f"shard_{self.split}_{self._shard_idx:04d}.bin"
            tokens.tofile(str(path))
            self._shard_idx += 1

        return {
            "total_tokens": self._total_tokens,
            "num_shards": self._shard_idx,
            "output_dir": str(self.output_dir)
        }
```

---

### 3.6 `HFHubPusher`

```python
# data/registry/hub_pusher.py

from huggingface_hub import HfApi, upload_folder, upload_file
from pathlib import Path

class HFHubPusher:
    """
    Pushes tokenizer and shard files to HuggingFace Hub.
    Acts as the artifact registry for all pipeline outputs.
    """

    def __init__(self, repo_id: str, token: str):
        self.repo_id = repo_id
        self.token = token
        self.api = HfApi()

    def create_repo(self, repo_type: str = "dataset") -> None:
        self.api.create_repo(
            repo_id=self.repo_id,
            token=self.token,
            repo_type=repo_type,
            exist_ok=True,
            private=False
        )

    def push_tokenizer(self, tokenizer_path: str) -> str:
        """
        Uploads tokenizer.json to HF Hub.
        Returns the URL of the uploaded file.
        """
        url = upload_file(
            path_or_fileobj=tokenizer_path,
            path_in_repo="tokenizer/tokenizer.json",
            repo_id=self.repo_id,
            token=self.token,
            repo_type="dataset"
        )
        return url

    def push_shards(self, shards_dir: str) -> None:
        """Uploads all .bin shards from a directory."""
        upload_folder(
            folder_path=shards_dir,
            repo_id=self.repo_id,
            token=self.token,
            repo_type="dataset",
            path_in_repo="shards/",
            ignore_patterns=["*.py", "*.yaml"]
        )
        print(f"Shards pushed to: https://huggingface.co/datasets/{self.repo_id}")
```

---

## 4. Pipeline Orchestrator

```python
# data/pipeline.py

import yaml
from ingestion.downloader import DatasetDownloader, DownloaderConfig
from cleaning.filters import (
    FilterPipeline, LengthFilter, WordLengthFilter,
    SymbolRatioFilter, BulletLinesFilter, AlphanumericFilter
)
from cleaning.deduplicator import MinHashDeduplicator
from tokenizer.trainer import BPETokenizerTrainer
from tokenizer.encoder import ShardEncoder
from registry.hub_pusher import HFHubPusher

def run_pipeline(config_path: str, hf_token: str):
    # ── Load Config ──────────────────────────────────
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    ic = cfg["ingestion"]
    cc = cfg["cleaning"]
    tc = cfg["tokenizer"]
    rc = cfg["registry"]

    # ── Step 1: Downloader ────────────────────────────
    print("=== STEP 1: Data Ingestion ===")
    downloader = DatasetDownloader(DownloaderConfig(
        source=ic["source"],
        subset=ic["subset"],
        num_samples=ic["num_samples"],
        text_column=ic["text_column"],
        streaming=ic["streaming"]
    )).load()

    # ── Step 2: Filter Pipeline ───────────────────────
    print("=== STEP 2: Filtering ===")
    filter_pipeline = FilterPipeline([
        LengthFilter(cc["min_length"], cc["max_length"]),
        WordLengthFilter(cc["min_avg_word_length"], cc["max_avg_word_length"]),
        SymbolRatioFilter(cc["max_symbol_to_word_ratio"]),
        BulletLinesFilter(cc["max_bullet_lines_ratio"]),
        AlphanumericFilter()
    ])
    deduplicator = MinHashDeduplicator(
        num_perm=cc["minhash_num_perm"],
        threshold=cc["minhash_threshold"]
    )

    def clean_stream():
        for i, text in enumerate(downloader.iterate()):
            if not filter_pipeline.apply(text):
                continue
            if deduplicator.is_duplicate(text, doc_id=str(i)):
                continue
            yield text

    # ── Step 3: Train Tokenizer ───────────────────────
    print("=== STEP 3: Tokenizer Training ===")
    trainer = BPETokenizerTrainer(
        vocab_size=tc["vocab_size"],
        min_frequency=tc["min_frequency"],
        special_tokens=tc["special_tokens"]
    ).train(clean_stream())
    trainer.save("artifacts/tokenizer.json")

    # ── Step 4: Encode Shards ─────────────────────────
    print("=== STEP 4: Shard Encoding ===")
    encoder = ShardEncoder(
        tokenizer=trainer,
        output_dir="artifacts/shards/",
        shard_size=tc["shard_size_tokens"]
    )
    stats = encoder.encode_stream(clean_stream())
    print(f"Encoding complete: {stats}")

    # ── Step 5: Push to HF Hub ────────────────────────
    print("=== STEP 5: Registry Push ===")
    pusher = HFHubPusher(repo_id=rc["hf_repo_id"], token=hf_token)
    pusher.create_repo()
    pusher.push_tokenizer("artifacts/tokenizer.json")
    if rc["push_shards"]:
        pusher.push_shards("artifacts/shards/")

    # ── Step 6: Final Report ──────────────────────────
    print("\n=== PIPELINE REPORT ===")
    print(filter_pipeline.rejection_report())
    print(f"Dedup rate: {deduplicator.dedup_rate:.2%}")
    print(f"Total tokens: {stats['total_tokens']:,}")
    print(f"Num shards: {stats['num_shards']}")
```

---

## 5. Data Contracts

| Stage               | Input                     | Output              | Format           |
| :------------------ | :------------------------ | :------------------ | :--------------- |
| Downloader          | HF dataset name + subset  | Raw text stream     | `Iterator[str]`  |
| FilterPipeline      | Raw text string           | Pass/fail bool      | `bool`           |
| MinHashDeduplicator | Text + doc_id             | Duplicate flag      | `bool`           |
| BPETokenizerTrainer | Text iterator             | Trained tokenizer   | `tokenizer.json` |
| ShardEncoder        | Text iterator + tokenizer | Binary token shards | `.bin` (uint16)  |
| HFHubPusher         | Local file paths          | HF Hub URLs         | Remote artifacts |

---

## 6. Testing Plan

```python
# tests/test_filters.py — example unit tests

def test_length_filter_rejects_short():
    f = LengthFilter(min_length=100)
    assert f.is_valid("short") == False

def test_length_filter_passes_valid():
    f = LengthFilter(min_length=10, max_length=1000)
    assert f.is_valid("a" * 50) == True

def test_pipeline_rejection_stats():
    pipeline = FilterPipeline([LengthFilter(min_length=1000)])
    pipeline.apply("short text")
    report = pipeline.rejection_report()
    assert report["total_passed"] == 0
    assert report["rejections_by_filter"]["length_filter"] == 1

def test_dedup_catches_near_duplicate():
    d = MinHashDeduplicator(threshold=0.8)
    text = "the quick brown fox jumps over the lazy dog " * 20
    assert d.is_duplicate(text, "doc_0") == False   # first time: not dup
    assert d.is_duplicate(text, "doc_1") == True    # second time: dup

def test_tokenizer_roundtrip():
    trainer = BPETokenizerTrainer(vocab_size=1000)
    trainer.train(iter(["hello world " * 1000]))
    ids = trainer.encode("hello world")
    assert trainer.decode(ids).strip() == "hello world"
```

---

## 7. Dependencies

```txt
# requirements.txt (Phase 1 only)
datasets>=2.20.0
huggingface_hub>=0.23.0
tokenizers>=0.19.0
datasketch>=1.6.5
numpy>=1.26.0
pyyaml>=6.0
tqdm>=4.66.0
```

---

## 8. Execution on Colab

```python
# In a Colab cell:
!pip install datasets huggingface_hub tokenizers datasketch pyyaml -q

import os
os.environ["HF_TOKEN"] = "hf_xxxx"  # from secrets

from data.pipeline import run_pipeline
run_pipeline("data/configs/data_config.yaml", hf_token=os.environ["HF_TOKEN"])
```

**Expected runtime on Colab free CPU:**

- Filtering 500K docs → ~15 min
- BPE tokenizer training (500K docs) → ~10 min
- Shard encoding → ~20 min
- HF Hub push → ~5 min
