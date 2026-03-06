## Project Overview

LLM Playground is an end-to-end system for building, training, evaluating, and serving Large Language Models from scratch. It uses only free-tier compute (Colab, Kaggle) and cloud services. The project is organized into phases:

- **Phase 1**: Data Layer (ingestion, cleaning, tokenization)
- **Phase 2**: Pre-Training Engine (GPT-2/Llama architecture)
- **Phase 3**: Post-Training (SFT, Reward Model, RLHF/GRPO)
- **Phase 4**: Evaluation Harness
- **Phase 5**: Chatbot UI & API

## Commands

### Running Tests

```bash
# Run all tests
pytest

# Run tests in a specific file
pytest data/tests/test_filters.py
pytest data/tests/test_downloader.py

# Run a specific test class
pytest data/tests/test_filters.py::TestLengthFilter

# Run a specific test
pytest data/tests/test_filters.py::TestLengthFilter::test_rejects_text_below_min_length
```

### Installing Dependencies

```bash
pip install -r requirements.txt
```

## Architecture

### Directory Structure

```
llm-playground/
├── data/                    # Phase 1: Data pipeline
│   ├── ingestion/           # DatasetDownloader - streams from HF datasets
│   ├── cleaning/            # FilterPipeline, MinHashDeduplicator
│   ├── tokenizer/           # BPETokenizerTrainer, ShardEncoder
│   ├── registry/            # HFHubPusher - uploads artifacts to HF Hub
│   ├── configs/             # YAML configuration (data_config.yaml)
│   └── tests/               # Unit tests for data modules
├── pretrain/                # Phase 2: Pre-training
│   └── model/               # GPT architecture, training loop, DataLoaderLite
├── posttrain/               # Phase 3: Post-training (SFT, RLHF)
├── eval/                    # Phase 4: Evaluation harness
├── serve/                   # Phase 5: API and UI
├── notebooks/               # Colab/Kaggle execution notebooks
├── docs/                    # HLD.md, LLD-phase 1.md
└── artifacts/               # Generated tokenizer, shards
```

### Data Pipeline Flow

```
Raw HF Dataset → DatasetDownloader → FilterPipeline → MinHashDeduplicator
    → BPETokenizerTrainer → ShardEncoder → HFHubPusher
```

### Key Design Patterns

1. **Streaming-first**: All data components use iterators/generators to avoid loading full datasets into RAM
2. **Dataclasses for config**: `DownloaderConfig`, `GPTConfig` use `@dataclass` for clean configuration
3. **Abstract base classes**: `BaseFilter` defines the interface; concrete filters (LengthFilter, etc.) implement it
4. **Short-circuit evaluation**: `FilterPipeline.apply()` stops on first filter failure for efficiency
5. **Stats tracking**: `FilterPipeline` tracks rejection counts per filter for observability

### Test Conventions

- Tests are located in `data/tests/`
- Each test file adds project root to `sys.path`:
  ```python
  sys.path.insert(0, str(Path(__file__).parent.parent))
  ```
- Test classes follow `TestClassName` naming (e.g., `TestLengthFilter`, `TestFilterPipeline`)
- Fixtures are minimal; most tests instantiate objects directly

## Configuration

All pipeline hyperparameters are in `data/configs/data_config.yaml`. No magic numbers in code. Key sections:

- `ingestion`: HF dataset source, subset, sample count
- `cleaning`: Filter thresholds, dedup settings
- `tokenizer`: Vocab size, special tokens, shard size
- `registry`: HF Hub repo ID and push flags

## Implementation Status

- **Phase 1 (Data Layer)**: In progress
  - `DatasetDownloader`: Complete with tests
  - `FilterPipeline` with 5 filters: Complete with tests
  - `MinHashDeduplicator`: Stub exists, needs implementation
  - `BPETokenizerTrainer`: Needs implementation
  - `ShardEncoder`: Needs implementation
  - `HFHubPusher`: Needs implementation
- **Phase 2 (Pre-train)**: Training script exists (`train_gpt2_5BT.py`), uses nanoGPT-style architecture
- **Phases 3-5**: Not yet implemented

## Pre-training Notes

The `pretrain/model/train_gpt2_5BT.py` is a complete GPT-2 training script that:

- Uses `torchrun` for DDP (multi-GPU) training
- Implements GPT-2 architecture from scratch (CausalSelfAttention, MLP, Block)
- Includes HellaSwag evaluation during training
- Saves checkpoints every 500 steps
- Expects tokenized data shards in `edu_fineweb5B/` directory
