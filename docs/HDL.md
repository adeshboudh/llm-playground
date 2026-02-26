# HLD: LLM Playground

**Document Version:** 1.0

**Date:** February 2026

**Author:** Adesh Boudh

**Status:** Draft

---

## 1. Overview

The LLM Playground is an end-to-end, modular system for building, training, evaluating, and serving Large Language Models from scratch. It covers the full lifecycle — data ingestion → pre-training → post-training (SFT + RLHF) → evaluation → chatbot serving — using only free-tier compute (Colab, Kaggle) and free-tier cloud services. The system is designed with production-grade separation of concerns so each module can be developed, replaced, or scaled independently.

---

## 2. Goals \& Non-Goals

**Goals:**

- Build every LLM pipeline stage from scratch with full understanding
- Design each component as an independent, testable service
- Stay within free-tier GPU/storage/hosting limits at all times
- Produce a deployable chatbot with a public API endpoint

**Non-Goals:**

- Training frontier-scale models (>7B params)
- Paid cloud infrastructure (AWS, GCP, Azure)
- Real-time high-concurrency serving (>100 req/s)

---

## 3. Success Metrics

| Metric                                   | Target                  |
| :--------------------------------------- | :---------------------- |
| Pre-training perplexity (val set)        | < 50 on 1B token subset |
| SFT model MT-Bench score                 | > 4.0 / 10              |
| Reward model accuracy (preference pairs) | > 65%                   |
| Eval harness (ARC-Easy)                  | > 55%                   |
| Chatbot UI uptime (HF Spaces)            | > 95%                   |
| End-to-end latency (inference)           | < 3s / response         |

---

## 4. System Architecture

### 4.1 High-Level Component Map

```
┌─────────────────────────────────────────────────────────────┐
│                     LLM PLAYGROUND                          │
│                                                             │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │  DATA    │   │  PRE-TRAIN   │   │   POST-TRAIN      │    │
│  │  LAYER   │──▶│   ENGINE     │──▶│   PIPELINE       │    │
│  └──────────┘   └──────────────┘   └───────────────────┘    │
│       │                │                     │              │
│       ▼                ▼                     ▼              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              ARTIFACT STORE (HF Hub)                 │   │
│  └──────────────────────────────────────────────────────┘   │
│       │                │                     │              │
│       ▼                ▼                     ▼              │
│  ┌──────────┐   ┌──────────────┐   ┌───────────────────┐    │
│  │  EVAL    │   │   MODEL      │   │   CHATBOT UI      │    │
│  │ HARNESS  │   │  REGISTRY    │   │   + API LAYER     │    │
│  └──────────┘   └──────────────┘   └───────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Data Flow Diagram

```
[Raw Web Data / HF Datasets]
         │
         ▼
[Ingestion & Sampling] ──→ [Quality Filters] ──→ [Cleaned Corpus]
                                                        │
                                                        ▼
                                              [BPE Tokenizer Training]
                                                        │
                                                        ▼
                                              [Tokenized .bin shards]
                                                        │
                                    ┌───────────────────┘
                                    ▼
                          [Pre-Training Loop]
                          (nanoGPT / Llama arch)
                                    │
                          [Base Model Checkpoint]
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
                [SFT]      [Reward Model]      [Eval Harness]
                    │               │
                    └──────┬────────┘
                           ▼
                     [PPO / GRPO Loop]
                           │
                    [Aligned Model]
                           │
                    [Inference Server]
                           │
                    [Chatbot UI / API]
```

---

## 5. Module Breakdown

### 5.1 Data Layer

| Sub-component     | Responsibility                            | Tools                   |
| :---------------- | :---------------------------------------- | :---------------------- |
| Ingestion         | Pull from FineWeb, Dolma, or custom crawl | `datasets`, `datatrove` |
| Quality Filter    | Length, dedup, perplexity-based filtering | `datasketch`, `kenlm`   |
| Tokenizer Trainer | Train BPE tokenizer on cleaned corpus     | HF `tokenizers`         |
| Shard Writer      | Write tokenized data to `.bin` shards     | `numpy`, custom writer  |
| Artifact Push     | Version and upload to HF Hub              | `huggingface_hub`       |

**Compute:** Colab CPU runtime (no GPU needed)

---

### 5.2 Pre-Training Engine

| Sub-component      | Responsibility                          | Tools                            |
| :----------------- | :-------------------------------------- | :------------------------------- |
| Model Architecture | GPT-2 / Llama-style Transformer         | `torch`, custom `nn.Module`      |
| Trainer            | Mixed precision, gradient checkpointing | `accelerate`, `deepspeed` ZeRO-2 |
| Data Loader        | Streaming from `.bin` shards            | Custom `IterableDataset`         |
| Checkpointing      | Save every N steps to HF Hub            | `huggingface_hub`                |
| Logging            | Loss, LR, grad norm per step            | `wandb`                          |

**Compute:** Kaggle 2×T4 (DDP) or P100 — 30 hrs/week free
**Model scale:** 117M – 350M parameters

---

### 5.3 Post-Training Pipeline

#### SFT Service

| Sub-component | Responsibility                                  | Tools                  |
| :------------ | :---------------------------------------------- | :--------------------- |
| Dataset Prep  | Format chat templates (ChatML / Llama-3 format) | `datasets`, `trl`      |
| LoRA Config   | QLoRA 4-bit fine-tuning                         | `peft`, `bitsandbytes` |
| SFT Trainer   | Supervised instruction tuning                   | `trl.SFTTrainer`       |

**Base model:** `SmolLM2-1.7B` or `Llama-3.2-1B` (fits T4 with QLoRA)

#### Reward Model Service

| Sub-component   | Responsibility                              | Tools                   |
| :-------------- | :------------------------------------------ | :---------------------- |
| Preference Data | Load `ultrafeedback_binarized`              | `datasets`              |
| RM Architecture | SFT base + scalar regression head           | `transformers`, `torch` |
| RM Trainer      | Bradley-Terry loss on chosen/rejected pairs | `trl.RewardTrainer`     |

#### RL Alignment Service

| Sub-component | Responsibility                                          | Tools             |
| :------------ | :------------------------------------------------------ | :---------------- |
| PPO Loop      | Classic RLHF with reward signal                         | `trl.PPOTrainer`  |
| GRPO Loop     | Group Relative Policy Optimization (verifiable rewards) | `trl.GRPOTrainer` |
| KL Controller | Prevent reward hacking via KL divergence penalty        | Built into `trl`  |

---

### 5.4 Artifact Store \& Model Registry

- **Storage:** HuggingFace Hub (free, unlimited public repos)
- **Versioning:** Git-based via `huggingface_hub` API
- **Naming convention:** `{username}/llm-playground-{stage}-{version}`
  - e.g., `you/llm-playground-sft-v1`, `you/llm-playground-rm-v1`
- **Experiment tracking:** W\&B free tier (run groups per phase)

---

### 5.5 Evaluation Harness

| Eval Type          | Method                                | Tool                        |
| :----------------- | :------------------------------------ | :-------------------------- |
| Perplexity         | Next-token prediction on held-out set | Custom eval loop            |
| Generation quality | BLEU, ROUGE on summarization tasks    | `evaluate` library          |
| Benchmark tasks    | ARC, HellaSwag, TruthfulQA, MMLU      | `lm-evaluation-harness`     |
| Human preference   | Pairwise chatbot arena voting         | Gradio + Supabase free tier |
| Leaderboard        | Score tracking per checkpoint         | W\&B Tables                 |

---

### 5.6 Chatbot UI \& API Layer

```
[User Browser]
     │  HTTPS
     ▼
[Gradio UI — HF Spaces (free)]
     │  REST / WebSocket
     ▼
[FastAPI Backend — Modal.com free tier]
     │  gRPC / HTTP
     ▼
[Inference Server — vLLM / llama.cpp]
  (Colab GPU tunneled via Cloudflare Tunnel)
     │
     ▼
[Model — loaded from HF Hub]
```

**Features:** Streaming output (SSE), conversation memory, system prompt control, temperature/top-p sliders

---

## 6. Infrastructure \& Compute Plan

| Phase           | Compute            | Service              | Cost |
| :-------------- | :----------------- | :------------------- | :--- |
| Data pipeline   | CPU, 12GB RAM      | Colab Free           | \$0  |
| Pre-training    | 2×T4 / P100        | Kaggle Free (30h/wk) | \$0  |
| SFT + RM        | T4 16GB            | Colab Free           | \$0  |
| GRPO/PPO        | P100 16GB          | Kaggle Free          | \$0  |
| Eval            | CPU                | Colab Free           | \$0  |
| Model storage   | Unlimited          | HF Hub Public        | \$0  |
| Experiment logs | Unlimited personal | W\&B Free            | \$0  |
| API backend     | Serverless         | Modal.com Free Tier  | \$0  |
| Chatbot UI      | CPU Space          | HF Spaces Free       | \$0  |
| Tunneling       | Unlimited          | Cloudflare Tunnel    | \$0  |

---

## 7. Repository Structure

```
llm-playground/
├── data/
│   ├── ingestion/          # Dataset download & sampling scripts
│   ├── cleaning/           # Quality filters, dedup
│   └── tokenizer/          # BPE tokenizer training
├── pretrain/
│   ├── model/              # Architecture (GPT, Llama)
│   ├── trainer/            # Training loop, DDP setup
│   └── configs/            # YAML configs per run
├── posttrain/
│   ├── sft/                # SFT dataset prep + trainer
│   ├── reward_model/       # RM architecture + trainer
│   └── rl/                 # PPO / GRPO loop
├── eval/
│   ├── perplexity.py
│   ├── benchmarks/         # lm-eval-harness integration
│   └── arena/              # Gradio human eval UI
├── serve/
│   ├── api/                # FastAPI inference server
│   ├── ui/                 # Gradio chatbot frontend
│   └── tunnel/             # Cloudflare tunnel config
├── notebooks/              # Colab/Kaggle execution notebooks
├── configs/                # Global YAML/TOML configs
├── tests/                  # Unit tests per module
├── docs/                   # HLD, LLD, ADRs
└── README.md
```

---

## 8. Key Design Decisions

| Decision             | Choice               | Rationale                               |
| :------------------- | :------------------- | :-------------------------------------- |
| Pre-training base    | nanoGPT → Llama arch | Start simple, evolve to SOTA            |
| Fine-tuning strategy | QLoRA 4-bit          | Only option viable on free T4           |
| RL method            | GRPO over PPO        | No value model → 2× memory savings      |
| Storage              | HF Hub               | Free, versioned, integrates natively    |
| Serving              | llama.cpp / vLLM     | Optimized for single GPU, low VRAM      |
| Orchestration        | Modal.com            | Serverless, free credits, Python-native |

---

## 9. Risks \& Mitigations

| Risk                                   | Mitigation                                                   |
| :------------------------------------- | :----------------------------------------------------------- |
| Colab session disconnects mid-training | Checkpoint every 500 steps to HF Hub                         |
| Kaggle weekly GPU quota exhausted      | Distribute across 2 Kaggle accounts or pause phase           |
| HF Hub rate limits                     | Cache artifacts locally in Google Drive                      |
| OOM on T4 during RLHF                  | Switch to GRPO; reduce batch size; use gradient accumulation |
| Free tier services deprecated          | All components are swappable; document alternatives          |

---

## 10. Milestones

| Phase      | Deliverable                                      | Target  |
| :--------- | :----------------------------------------------- | :------ |
| Phase 1    | Cleaned dataset + BPE tokenizer pushed to HF Hub | Week 2  |
| Phase 2    | Pre-trained 125M model, W\&B loss curves         | Week 4  |
| Phase 3a   | SFT model checkpoint (QLoRA merged)              | Week 6  |
| Phase 3b/c | RM + GRPO-aligned model                          | Week 8  |
| Phase 4    | Eval harness report (perplexity + ARC score)     | Week 9  |
| Phase 5    | Live chatbot on HF Spaces with public API        | Week 10 |
