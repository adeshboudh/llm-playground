## LLM Playground

LLM Playground is an end-to-end system for building, training, evaluating, and serving Large Language Models from scratch. It uses only free-tier compute (Colab, Kaggle) and cloud services. The project is organized into phases:

- **Phase 1**: Data Layer (ingestion, cleaning, tokenization)
- **Phase 2**: Pre-Training Engine (GPT-2 architecture)
- **Phase 3**: Post-Training (SFT, Reward Model, GRPO)
- **Phase 4**: Evaluation Harness
- **Phase 5**: Chatbot UI & API

**Current Status**: All phases are fully implemented with working code and tests.

## Quick Start

### Try the Pre-trained Chatbot

The project includes a pre-trained GPT-2 model (124M parameters) fine-tuned with GRPO. To run the playground:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the FastAPI server
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

# Open http://localhost:8000 in your browser
# Enter prompts and generate responses with adjustable temperature
```

The model is loaded from `model/gpt2_grpo_merged/` and uses a chat template with `<|user|>` and `<|assistant|>` tokens.

### Run Tests

```bash
pytest                    # All tests
pytest tests/test_gpt.py  # Specific module
```

### Train Your Own Model

Follow the pipeline order below or see individual scripts in:
- Data preparation: `data/pipeline.py`
- Pre-training: `model/train.py`
- Post-training: `posttrain/sft/`, `posttrain/reward_model/`, `posttrain/rl/`

---

## Commands

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest

# Run tests in a specific file
pytest tests/test_downloader.py
pytest tests/test_filters.py
pytest tests/test_deduplicator.py
pytest tests/test_encoder.py
pytest tests/test_tokenizer_trainer.py
pytest tests/test_pipeline.py
pytest tests/test_gpt.py
pytest tests/test_dataloader.py

# Run a specific test class
pytest tests/test_filters.py::TestLengthFilter

# Run a specific test
pytest tests/test_filters.py::TestLengthFilter::test_rejects_text_below_min_length
```

### Running the Chatbot Playground

```bash
# Start the FastAPI server with auto-reload
uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

# Then open http://localhost:8000 in your browser
```

### Running Training Pipelines

```bash
# Phase 2: Pre-train GPT-2
python model/train.py

# Phase 3a: SFT with LoRA
python posttrain/sft/tokenize_ultrachat.py
python posttrain/sft/train_sft_lora.py
python posttrain/sft/infer_sft.py

# Phase 3b: Train Reward Model
python posttrain/reward_model/prepare_rm_data.py
python posttrain/reward_model/train_rm.py
python posttrain/reward_model/infer_rm.py

# Phase 3c: GRPO Alignment
python posttrain/rl/train_grpo_gpu.py

# Note: To serve the model, either use the pre-merged model in model/gpt2_grpo_merged/
# or manually merge LoRA weights with the base model using PEFT's merge_and_unload()
```

## Architecture

### Directory Structure

```
llm-playground/
├── data/                      # Phase 1: Data pipeline
│   ├── ingestion/             # DatasetDownloader - streams from HF datasets
│   ├── cleaning/              # FilterPipeline, MinHashDeduplicator
│   ├── tokenizer/             # BPETokenizerTrainer, ShardEncoder
│   ├── registry/              # HFHubPusher - uploads artifacts to HF Hub
│   ├── pipeline.py            # Main orchestrator for data pipeline
│   └── configs/               # YAML configuration files
├── model/                     # Core model components
│   ├── gpt.py                 # GPT-2 architecture implementation
│   ├── train.py               # Pre-training script
│   ├── dataloader.py          # DataLoaderLite for streaming shards
│   ├── eval/                  # Evaluation modules (HellaSwag, etc.)
│   ├── configs/               # Model configuration YAMLs
│   ├── scripts/               # Utility scripts
│   └── gpt2_grpo_merged/      # Merged model checkpoint (SFT + GRPO)
├── checkpoints/               # Training checkpoints
│   ├── pretrain/              # Pre-training checkpoints
│   ├── sft/                   # SFT LoRA adapters
│   ├── reward_model/          # Reward model checkpoints
│   └── rl/                    # GRPO training checkpoints
├── posttrain/                 # Phase 3: Post-training
│   ├── sft/                   # SFT with LoRA
│   │   ├── tokenize_ultrachat.py
│   │   ├── train_sft_lora.py
│   │   ├── infer_sft.py
│   │   └── config.py
│   ├── reward_model/          # Reward model training
│   │   ├── prepare_rm_data.py
│   │   ├── train_rm.py
│   │   ├── infer_rm.py
│   │   └── config.py
│   └── rl/                    # GRPO alignment
│       ├── grpo.py
│       ├── train_grpo_gpu.py
│       └── config.py
├── serve/                     # Phase 5: API and UI
│   ├── app.py                 # FastAPI server with streaming
│   └── templates/             # HTML templates for chatbot UI
├── tests/                     # Root-level unit tests
├── notebooks/                 # Colab/Kaggle execution notebooks
├── docs/                      # HLD.md, LLD-phase 1.md
└── artifacts/                 # Generated artifacts (tokenizer, shards)
```

## Stack

| Phase    | Component                          | Details                                          |
| -------- | ---------------------------------- | ------------------------------------------------ |
| Data     | FineWeb-edu + BPE tokenizer        | Custom filters, MinHash dedup, HF Hub push      |
| Pretrain | GPT-2 124M from scratch            | DDP training on tokenized shards                |
| SFT      | LoRA fine-tuning                   | UltraChat 200k, r=16, alpha=32, TPU/XLA training  |
| RM       | Reward Model                       | GPT-2 base + scalar head, trained on preferences |
| GRPO     | Group Relative Policy Optimization | G=4, KL-anchored, verifiable rewards            |
| Eval     | HellaSwag, custom eval             | Perplexity, generation quality                  |
| Serve    | FastAPI + streaming                | Browser chatbot playground with SSE            |

## Pipeline Order

```
1. data/pipeline.py                     → Ingest, filter, tokenize FineWeb-edu
2. model/train.py                       → Pre-train GPT-2 from scratch
3. posttrain/sft/tokenize_ultrachat.py  → Prepare UltraChat dataset
4. posttrain/sft/train_sft_lora.py      → Fine-tune with LoRA
5. posttrain/reward_model/train_rm.py   → Train reward model on preferences
6. posttrain/rl/train_grpo_gpu.py       → GRPO alignment
7. serve/app.py                         → FastAPI playground (merged model)
```

### Key Design Patterns

1. **Streaming-first**: All data components use iterators/generators to avoid loading full datasets into RAM
2. **Dataclasses for config**: `DownloaderConfig`, `GPTConfig` use `@dataclass` for clean configuration
3. **Abstract base classes**: `BaseFilter` defines the interface; concrete filters (LengthFilter, etc.) implement it
4. **Short-circuit evaluation**: `FilterPipeline.apply()` stops on first filter failure for efficiency
5. **Stats tracking**: `FilterPipeline` tracks rejection counts per filter for observability

### Test Conventions

- Tests are located in `tests/` (root-level directory)
- Test files cover all major components: downloader, filters, deduplicator, encoder, tokenizer trainer, pipeline, GPT model, dataloader
- Test classes follow `TestClassName` naming (e.g., `TestLengthFilter`, `TestFilterPipeline`, `TestDatasetDownloader`)
- Minimal fixtures; most tests instantiate objects directly for unit testing
- Run with `pytest` from project root

## Configuration

All hyperparameters are centralized in YAML config files. No magic numbers in code.

### Data Pipeline Config (`data/configs/data_config.yaml`)

- `ingestion`: HF dataset source (`HuggingFaceFW/fineweb-edu`), subset, sample count
- `cleaning`: Filter thresholds (length, word length, symbol ratio, bullet lines, alphanumeric), dedup settings
- `tokenizer`: Vocab size (32k), special tokens, shard size (100M tokens)
- `registry`: HF Hub repo ID and push flags
- `artifacts`: Output paths for tokenizer and shards

### Model & Training Config (`model/configs/model_config.yaml`)

- `model`: Architecture params (GPT-2: 768 dim, 12 heads, 12 layers, 50k vocab, 1024 context)
- `training`: Batch sizes, learning rates, max steps (5.6B tokens target), checkpoint intervals
- `paths`: Shard directory, log directory
- `logging`: Weights & Biases project and run name

## Pre-training Notes

The `model/train.py` is a complete GPT-2 training script that:

- Uses `torchrun` for DDP (multi-GPU) training
- Implements GPT-2 architecture from scratch (CausalSelfAttention, MLP, Block)
- Includes HellaSwag evaluation during training
- Saves checkpoints every 500 steps
- Expects tokenized data shards in `artifacts/shards/` directory
- Configurable via `model/configs/model_config.yaml`

## Implementation Status

All phases are fully implemented and functional:

### Phase 1: Data Layer ✓ COMPLETE
- `DatasetDownloader`: Streams from HuggingFace datasets, supports sampling
- `FilterPipeline`: 5 quality filters with short-circuit evaluation
- `MinHashDeduplicator`: MinHash-based deduplication (disabled for FineWeb-edu)
- `BPETokenizerTrainer`: BPE tokenizer training on cleaned corpus
- `ShardEncoder`: Tokenized shard writer (`.bin` format)
- `HFHubPusher`: Upload artifacts to HuggingFace Hub
- **Tests**: All data modules have unit tests in `tests/`

### Phase 2: Pre-Training ✓ COMPLETE
- GPT-2 architecture from scratch (`model/gpt.py`)
- DDP-ready training loop (`model/train.py`)
- DataLoaderLite for streaming `.bin` shards
- HellaSwag evaluation during training
- Checkpointing to `checkpoints/pretrain/`
- Sample checkpoint: `checkpoints/pretrain/model_05000.pt`

### Phase 3: Post-Training ✓ COMPLETE
- **SFT** (Supervised Fine-Tuning):
  - UltraChat 200k dataset tokenization
  - LoRA configuration (r=16, alpha=32, dropout=0.05)
  - Training script with PEFT (`posttrain/sft/train_sft_lora.py`)
  - Inference script (`posttrain/sft/infer_sft.py`)
  - Checkpoints: `checkpoints/sft/adapter_sft_v1.pt`

- **Reward Model**:
  - GPT-2 base + scalar regression head
  - Preference pair training (hh-rlhf format)
  - Training (`posttrain/reward_model/train_rm.py`) and inference (`infer_rm.py`)
  - Checkpoints: `checkpoints/reward_model/`

- **GRPO** (Group Relative Policy Optimization):
  - Verifiable reward system (no value model needed)
  - Group size G=4, KL-anchored regularization
  - Training script: `posttrain/rl/train_grpo_gpu.py`
  - Merged model: `model/gpt2_grpo_merged/` (SFT + GRPO weights)

### Phase 4: Evaluation ✓ COMPLETE
- HellaSwag evaluation (`model/eval/hellaswag.py`)
- Perplexity calculation on validation sets
- Custom eval integration in pre-training loop
- Result logging and reporting

### Phase 5: Serving ✓ COMPLETE
- FastAPI inference server (`serve/app.py`)
- Streaming response support (SSE)
- Browser-based chatbot UI (HTML + Jinja2 templates)
- Model loading from `model/gpt2_grpo_merged/`
- Temperature and max tokens controls
- `uvicorn` server with auto-reload support
