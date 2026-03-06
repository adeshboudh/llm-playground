# Data Pipeline

The data pipeline is responsible for ingesting raw web text, cleaning it, deduplicating it, training a BPE tokenizer, and encoding the output into binary shards ready for model training.

---

## Pipeline Overview

```
HuggingFace Dataset
        │
        ▼
┌─────────────────┐
│   Downloader    │  Streams N docs from HuggingFaceFW/fineweb-edu
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Filter Pipeline │  Rejects noisy documents via 5 filters
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deduplicator   │  MinHash LSH — removes near-duplicate documents
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BPE Tokenizer   │  Trains a GPT-2 style byte-level BPE tokenizer
│    Trainer      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Shard Encoder  │  Encodes clean docs into uint16 binary shards
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   HF Hub Push   │  Uploads tokenizer + shards to HuggingFace Hub
└─────────────────┘
```

---

## Directory Structure

```
data/
├── configs/
│   └── data_config.yaml        # Single source of truth for all pipeline parameters
├── ingestion/
│   └── downloader.py           # DatasetDownloader — wraps HuggingFace datasets
├── cleaning/
│   ├── filters.py              # FilterPipeline + 5 individual filter classes
│   └── deduplicator.py         # MinHashDeduplicator — LSH-based near-dedup
├── tokenizer/
│   └── trainer.py              # BPETokenizerTrainer — trains and saves tokenizer
├── encoding/
│   └── encoder.py              # ShardEncoder — writes uint16 binary shard files
├── registry/
│   └── hub_pusher.py           # HFHubPusher — pushes artifacts to HF Hub
└── pipeline.py                 # run_pipeline() — orchestrates all steps
```

---

## Configuration

All parameters live in `data/configs/data_config.yaml`. Never hardcode values in pipeline code.

```yaml
ingestion:
  source: HuggingFaceFW/fineweb-edu
  subset: sample-10BT
  num_samples: 50000
  text_column: text
  streaming: true

cleaning:
  min_length: 100
  max_length: 100000
  min_avg_word_length: 3
  max_avg_word_length: 10
  max_symbol_to_word_ratio: 0.35 # Calibrated on real FineWeb-Edu data
  max_bullet_lines_ratio: 0.9
  minhash_num_perm: 128
  minhash_threshold: 0.85

tokenizer:
  vocab_size: 32000
  min_frequency: 2
  special_tokens:
    - <|endoftext|>
    - <|pad|>
    - <|unk|>
    - <|im_start|>
    - <|im_end|>
  shard_size_tokens: 100000

artifacts:
  tokenizer_path: artifacts/tokenizer
  shards_dir: artifacts/shards

registry:
  hf_repo_id: your-username/your-dataset-repo
  push_shards: false
```

---

## Components

### Downloader (`data/ingestion/downloader.py`)

Wraps HuggingFace `datasets` in streaming mode. Yields raw text strings one at a time — no full dataset loaded into memory.

**Key design decision:** Streaming mode means the pipeline can handle arbitrarily large datasets without RAM constraints.

---

### Filter Pipeline (`data/cleaning/filters.py`)

Applies 5 sequential filters. A document is rejected if **any** filter fails.

| Filter               | What it rejects                                              | Config key                                   |
| -------------------- | ------------------------------------------------------------ | -------------------------------------------- |
| `LengthFilter`       | Documents shorter or longer than character bounds            | `min_length`, `max_length`                   |
| `WordLengthFilter`   | Docs with avg word length outside bounds (catches gibberish) | `min_avg_word_length`, `max_avg_word_length` |
| `SymbolRatioFilter`  | Docs where symbols-per-word exceeds threshold                | `max_symbol_to_word_ratio`                   |
| `BulletLinesFilter`  | Docs that are mostly bullet points / list dumps              | `max_bullet_lines_ratio`                     |
| `AlphanumericFilter` | Docs with too few alphanumeric characters                    | hardcoded                                    |

**Calibration note:** `max_symbol_to_word_ratio` was empirically set to `0.35` after sampling real FineWeb-Edu documents. Real English prose has a natural symbol ratio of 0.15–0.32 (punctuation, apostrophes, quotes). The original value of `0.1` rejected 96% of valid documents.

---

### Deduplicator (`data/cleaning/deduplicator.py`)

Uses MinHash + LSH (Locality Sensitive Hashing) to detect near-duplicate documents.

- `num_perm=128` — number of hash permutations (higher = more accurate, slower)
- `threshold=0.85` — Jaccard similarity threshold above which two docs are considered duplicates
- State persists across the entire pipeline run — a document seen in any pass is remembered
- `reset()` clears all state — use only when starting a completely fresh pipeline run

**Verified behavior:**

- Original document: not flagged ✅
- Near-duplicate (one sentence changed): flagged ✅
- Completely different document: not flagged ✅

**Note:** FineWeb-Edu is pre-deduplicated at source. Expect 0% dedup rate on this dataset. The deduplicator is critical when mixing multiple data sources.

---

### BPE Tokenizer Trainer (`data/tokenizer/trainer.py`)

Trains a GPT-2 style Byte-Level BPE tokenizer from scratch using HuggingFace `tokenizers`.

- Normalizer: NFC unicode normalization
- Pre-tokenizer: ByteLevel (no prefix space)
- Decoder: ByteLevel
- `save(path)` — if path has no extension, saves `tokenizer.json` inside it as a directory
- `load(path)` — if path is a directory, loads `tokenizer.json` from inside it

**Windows note:** The Rust-backed `tokenizers` library requires forward-slash paths. Always use `.as_posix()` when passing paths to `save()` or `load()`.

---

### Shard Encoder (`data/encoding/encoder.py`)

Encodes tokenized documents into binary shard files for efficient training data loading.

- Output format: `uint16` numpy arrays saved as `.bin` files
- Shard naming: `shard_train_XXXX.bin`
- Each shard contains up to `shard_size_tokens` tokens
- Token count per shard is logged at write time

---

### HF Hub Pusher (`data/registry/hub_pusher.py`)

Uploads pipeline artifacts (tokenizer + shards) to a HuggingFace Hub dataset repository.

- Creates the repo if it doesn't exist
- Skips push if files are unchanged (HF Hub deduplication)
- Set `push_shards: false` in config during development to avoid unnecessary uploads

---

## Running the Pipeline

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Run full pipeline
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from data.pipeline import run_pipeline
run_pipeline('data/configs/data_config.yaml')
"
```

### Sample Output

```
=== PIPELINE REPORT ===
{'total_seen': 50000, 'total_passed': 48514, 'pass_rate': 0.9703,
'rejections_by_filter': {'length_filter': 49, 'word_length_filter': 3,
'symbol_ratio_filter': 1352, 'bullet_lines_filter': 82, 'alphanum_filter': 0}}
Dedup rate: 0.03%
Total tokens: 49,357,932
Num shards: 1
```

```text
Tokenizer vocab size: 32,000 (trained on ~50k docs / ~49M tokens)
```

---

## Tests

```bash
# Run all pipeline tests
pytest tests/test_pipeline.py -v

# Run full test suite
pytest --tb=short -q
```

### Test Coverage

| Test file                         | What it covers                                             |
| --------------------------------- | ---------------------------------------------------------- |
| `tests/test_filters.py`           | Each filter individually — boundary conditions, edge cases |
| `tests/test_deduplicator.py`      | LSH state, near-dup detection, reset behavior              |
| `tests/test_tokenizer_trainer.py` | Train, encode, decode, save, load                          |
| `tests/test_pipeline.py`          | End-to-end pipeline — shard output, token count, pass rate |

---

## Known Constraints

- **Single-pass design:** The pipeline processes documents once and feeds the same `clean_docs` list to both the tokenizer trainer and shard encoder. Do not reintroduce a streaming `clean_stream()` pattern without making the deduplicator stateless or scoped per pass.
- **Memory:** All clean documents are held in memory during the pipeline run. At `num_samples=100k+`, monitor RAM usage and consider chunked shard writing.
- **Windows paths:** Any library with a Rust or C++ backend (tokenizers, safetensors) requires forward-slash paths. Use `.as_posix()` when constructing paths passed to these libraries.
