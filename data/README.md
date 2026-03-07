# Data Pipeline

This document explains the end-to-end data pipeline for the LLM Playground project —
what each component does, why it was built the way it was, and the decisions made
during development and debugging.

---

## Overview

The pipeline takes raw web text from HuggingFace datasets, cleans and deduplicates it,
trains a BPE tokenizer from scratch on the cleaned corpus, encodes the corpus into
binary token shards, and pushes everything to the HuggingFace Hub.

```

HuggingFace Dataset
↓
Step 1: Ingestion         — Download raw documents (streaming)
↓
Step 2: Filter + Dedup    — Remove low-quality and near-duplicate documents
↓
Step 3: Tokenizer         — Train a BPE tokenizer on clean documents
↓
Step 4: Shard Encoding    — Encode documents into uint16 binary shards
↓
Step 5: Registry Push     — Upload tokenizer + shards to HuggingFace Hub

```

Everything is driven by a single config file: `data/configs/data_config.yaml`.

---

## Directory Structure

```

data/
├── configs/
│   └── data_config.yaml        \# Single source of truth for all pipeline params
├── ingestion/
│   └── downloader.py           \# Streaming dataset downloader
├── cleaning/
│   ├── filters.py              \# FilterPipeline + individual filter classes
│   └── deduplicator.py         \# MinHash LSH deduplicator
├── tokenizer/
│   └── trainer.py              \# BPETokenizerTrainer (train, save, load, encode)
├── encoding/
│   └── encoder.py              \# ShardEncoder — writes uint16 .bin shard files
├── registry/
│   └── hub_pusher.py           \# HuggingFace Hub upload
├── pipeline.py                 \# Orchestrates all steps end-to-end
└── README.md                   \# This file

tests/
├── test_filters.py
├── test_deduplicator.py
├── test_tokenizer_trainer.py
└── test_pipeline.py            \# Integration tests for full pipeline

scratch/
├── diagnose_filter.py          \# One-off: calibrated SymbolRatioFilter threshold
├── stress_dedup.py             \# One-off: verified deduplicator at 1000-doc scale
└── check_tokenizer_output.py   \# One-off: verified trainer.save() file output

```

---

## Config File

**`data/configs/data_config.yaml`** controls everything. No hardcoded values in code.

Key sections:

```yaml
ingestion:
  source: HuggingFaceFW/fineweb-edu # Dataset on HF Hub
  subset: sample-10BT # 10B token sample
  num_samples: 100 # How many docs to pull per run
  text_column: text
  streaming: true # Never downloads full dataset to disk

cleaning:
  min_length: 100 # Characters
  max_length: 100000
  min_avg_word_length: 3.0
  max_avg_word_length: 10.0
  max_symbol_to_word_ratio: 0.35 # Calibrated — see Filter Calibration below
  max_bullet_lines_ratio: 0.8
  minhash_num_perm: 128
  minhash_threshold: 0.85

tokenizer:
  vocab_size: 32000
  min_frequency: 2
  special_tokens: [<|endoftext|>, <|pad|>, <|unk|>, <|im_start|>, <|im_end|>]
  shard_size_tokens: 100000000 # 100M tokens per shard

artifacts:
  tokenizer_path: artifacts/tokenizer
  shards_dir: artifacts/shards

registry:
  hf_repo_id: adesh01/llm-playground-data-v1
  push_shards: false # Set true for full runs
```

---

## Step 1 — Ingestion (`data/ingestion/downloader.py`)

**What it does:** Downloads documents from a HuggingFace dataset using streaming mode.

**Why streaming:** FineWeb-Edu sample-10BT is ~500GB. Streaming means documents are
pulled one at a time over the network — no local disk required. The downloader wraps
HuggingFace `datasets.load_dataset(..., streaming=True)` and yields plain text strings.

**Key design:** The downloader is initialized once and iterated once. Creating multiple
downloader instances for the same config produces independent streams starting from the
beginning — this was critical for avoiding a double-pass deduplication bug (see Bugs section).

---

## Step 2 — Filtering (`data/cleaning/filters.py`)

**What it does:** Runs each document through a chain of quality filters. Any document
failing any filter is rejected. Rejection counts are tracked per filter for reporting.

### Filters

| Filter               | What it rejects                              | Threshold           |
| :------------------- | :------------------------------------------- | :------------------ |
| `LengthFilter`       | Docs too short or too long (character count) | 100–100,000 chars   |
| `WordLengthFilter`   | Docs with abnormal average word length       | 3.0–10.0 chars/word |
| `SymbolRatioFilter`  | Docs with too many symbols relative to words | max 0.35 ratio      |
| `BulletLinesFilter`  | Docs that are mostly bullet points / lists   | max 0.8 ratio       |
| `AlphanumericFilter` | Docs with insufficient alphanumeric content  | built-in threshold  |

### Filter Calibration — Why `max_symbol_to_word_ratio: 0.35`

The initial threshold was `0.1` (from the LLD). This was never validated against real data.
When run against FineWeb-Edu, it rejected 96% of documents because normal English prose
naturally contains punctuation (periods, commas, apostrophes, quotes) at ratios of 0.15–0.20.

A diagnostic script (`scratch/diagnose_filter.py`) was run on 10 real documents:

```
words= 633  symbols= 109  ratio=0.17  passed=False  ← clean article, wrong rejection
words= 860  symbols= 143  ratio=0.17  passed=False  ← clean article, wrong rejection
words=2762  symbols= 518  ratio=0.19  passed=False  ← clean article, wrong rejection
```

The threshold was raised to `0.35` after measuring real document ratios. At this threshold:

- Normal prose (ratio 0.12–0.25) passes
- Genuinely noisy documents (symbol-heavy spam, code dumps) at ratio > 0.35 are rejected
- Verified pass rate: ~96% on FineWeb-Edu

**Lesson:** Every filter threshold must be calibrated on a sample of actual target data
before use. Never copy thresholds from design documents without empirical validation.

---

## Step 2 (cont.) — Deduplication (`data/cleaning/deduplicator.py`)

**What it does:** Removes near-duplicate documents using MinHash LSH (Locality Sensitive
Hashing). Two documents are considered near-duplicates if their Jaccard similarity
exceeds the configured threshold (default: 0.85).

**Why MinHash LSH:** Exact deduplication (hashing) only catches identical documents.
Web text often has near-duplicates — same article with minor edits, scraped multiple times
with different boilerplate. MinHash estimates Jaccard similarity in O(1) per document
using `num_perm=128` hash permutations and an LSH index.

**How it works:**

1. Document is split into character n-grams
2. MinHash signature (128 hash values) computed
3. LSH checks if any existing document has similar signature
4. If similarity > threshold: reject as duplicate, else: add to index

**Dedup rate on FineWeb-Edu:** 0.00% at 1000 docs — expected, because FineWeb-Edu is
already deduplicated by its creators before publishing. The deduplicator was verified
using injected near-duplicates (see `scratch/stress_dedup.py`):

```
Original doc flagged as dup:   False  ✅
Near-duplicate flagged as dup: True   ✅
Different doc flagged as dup:  False  ✅
```

### `reset()` Method

`reset()` clears the LSH index and all counters:

```python
def reset(self) -> None:
    self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
    self._seen_count = 0
    self._dup_count = 0
```

**Important:** Do not call `reset()` between pipeline passes. The deduplicator must
maintain state across the entire corpus. Calling `reset()` mid-pipeline causes pass 2
to treat all previously seen documents as new (0% dedup rate) — this was a real bug
that was caught and fixed.

---

## Step 3 — Tokenizer Training (`data/tokenizer/trainer.py`)

**What it does:** Trains a GPT-2 style Byte-Level BPE tokenizer from scratch on the
cleaned corpus. Does not use a pretrained vocabulary.

**Architecture choices:**

- **Byte-Level BPE:** Operates on raw bytes, never produces `<unk>` for any input
- **NFC Normalization:** Unicode normalization before tokenization for consistency
- **ByteLevel pre-tokenizer:** Splits on spaces and encodes bytes as visible characters
- **Special tokens:** `<|endoftext|>`, `<|pad|>`, `<|unk|>`, `<|im_start|>`, `<|im_end|>`

**`save()` and `load()` design:**

`save(path)` accepts either a file path (`artifacts/tokenizer/tokenizer.json`) or a
directory stem (`artifacts/tokenizer`). If no file extension is present, it automatically
appends `/tokenizer.json`. It returns the resolved file path so callers don't have to
reconstruct it.

```python
saved_path = trainer.save(cfg["artifacts"]["tokenizer_path"])
# saved_path → PosixPath('artifacts/tokenizer/tokenizer.json')
```

**Windows gotcha:** The `tokenizers` library has a Rust backend that does not normalize
Windows backslash paths. All paths passed to `tokenizer.save()` and `Tokenizer.from_file()`
must use forward slashes (`.as_posix()`). Failing to do this raises:
`ValueError: Provided path is not a file on the local file system`

---

## Step 4 — Shard Encoding (`data/encoding/encoder.py`)

**What it does:** Encodes each clean document into token IDs and writes them to binary
`.bin` files (shards). Each token is stored as a `uint16` (2 bytes), matching the format
used by GPT-2 / nanoGPT training pipelines.

**Why shards:** LLM training reads data sequentially in large chunks. Storing tokens in
flat binary files enables memory-mapped I/O (`np.memmap`) during training — no Python
overhead, no HuggingFace dataset loading. This is the standard approach for training
at scale (used by FineWeb, RedPajama, Dolma).

**Shard naming:** `shard_train_0000.bin`, `shard_train_0001.bin`, etc.

---

## Step 5 — Registry Push (`data/registry/hub_pusher.py`)

**What it does:** Uploads the trained tokenizer and encoded shards to HuggingFace Hub
for versioned storage and team access.

**Note:** During tests, `push_shards: false` is set in config to avoid pushing test
artifacts to the Hub. The tokenizer is still pushed (idempotent — HF skips if unchanged).

---

## Pipeline Orchestration (`data/pipeline.py`)

**Critical design: Single-Pass Architecture**

The pipeline collects all clean documents into memory once, then reuses the list for
both tokenizer training and shard encoding:

```python
# Single pass — filter + dedup once
clean_docs = []
for i, text in enumerate(downloader.iterate()):
    if not filter_pipeline.apply(text):
        continue
    if deduplicator.is_duplicate(text, doc_id=str(i)):
        continue
    clean_docs.append(text)

# Reuse the same list for both downstream steps
trainer.train(iter(clean_docs))
encoder.encode_stream(iter(clean_docs))
```

**Why not a generator / lazy stream?**

An earlier design used `clean_stream()` — a generator function that re-instantiated the
downloader and re-ran filtering/dedup on each call. This was broken because:

1. The deduplicator is stateful — pass 1 added all documents to the LSH index
2. Pass 2 (shard encoding) saw all documents as duplicates → 0 tokens written
3. The generator looked clean but violated the contract of shared stateful components

**Rule:** Stateful components (deduplicators, counters, caches) cannot be shared across
independent generator invocations. Materialize the data once when the dataset fits in memory.

---

## Running the Pipeline

```bash
# Dry run (100 docs, no shard push)
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from data.pipeline import run_pipeline
run_pipeline('data/configs/data_config.yaml')
"

# Full test suite
pytest --tb=short -q

# Stress test deduplicator at 1000 docs
PYTHONPATH=. python scratch/stress_dedup.py
```

**Expected pipeline report (100 docs):**

```
total_seen:   100
total_passed: ~96
pass_rate:    0.96
symbol_ratio_filter: ~4 rejections
Dedup rate:   0.00%   ← expected on FineWeb-Edu (pre-deduplicated)
Total tokens: ~80,000
Num shards:   1
```

---

## Tests

```
tests/test_filters.py           — Unit tests for each filter class
tests/test_deduplicator.py      — Unit tests for MinHashDeduplicator + reset()
tests/test_tokenizer_trainer.py — Unit tests for train/save/load/encode
tests/test_pipeline.py          — Integration tests: end-to-end pipeline on 20 docs
```

Run with:

```bash
pytest --tb=short -q
# Expected: 66 passed
```

---

## Known Gotchas

| Issue                               | Root Cause                                                | Fix                                                      |
| :---------------------------------- | :-------------------------------------------------------- | :------------------------------------------------------- |
| 96% filter rejection rate           | `max_symbol_to_word_ratio: 0.1` too strict                | Calibrate on real data → use `0.35`                      |
| 0 tokens in shards                  | Shared deduplicator across two generator passes           | Single-pass architecture                                 |
| `ValueError: not a file` on Windows | Rust tokenizers backend rejects backslash paths           | Use `.as_posix()` for all paths                          |
| `tokenizer.json` not found          | `save()` writing file named `tokenizer` with no extension | Check suffix, append `/tokenizer.json` if directory stem |
| Pytest collecting `scratch/`        | No `testpaths` configured                                 | Set `testpaths = ["tests"]` in `pyproject.toml`          |
