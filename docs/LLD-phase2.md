# LLD — Phase 2: GPT-2 110M Pretraining

## Objective

Train a GPT-2 110M model from scratch on 5B tokens of cleaned FineWeb-Edu data using a custom BPE tokenizer (vocab=32,000), with W&B loss tracking, HellaSwag evaluation, and checkpoint export to HuggingFace Hub.

---

## Delta from Previous Training Run

| Aspect             | Previous (`train_gpt2.py`)      | This Run                                 |
| ------------------ | ------------------------------- | ---------------------------------------- |
| Tokenizer          | `tiktoken` GPT-2 (vocab=50,257) | Our `BPETokenizerTrainer` (vocab=32,000) |
| Vocab size (model) | 50,304 (padded)                 | 32,064 (padded to multiple of 64)        |
| Data               | Raw FineWeb-Edu, no cleaning    | Cleaned + deduped pipeline shards        |
| Loss tracking      | `log.txt` only                  | W&B + `log.txt`                          |
| Checkpoint format  | `.pt` (raw state dict)          | `safetensors` + metadata JSON            |
| Token budget       | ~3B (cut off at step 5000)      | 5B (full run)                            |
| Warmup steps       | 715                             | 2,000                                    |
| Max steps          | 19,073                          | ~9,537 at 0.5M tok/step batch            |

---

## Directory Structure

```
model/
├── __init__.py
├── gpt.py                    # Architecture: CausalSelfAttention, MLP, Block, GPT
├── train.py                  # Training loop, W&B, checkpointing
├── dataloader.py             # DataLoaderLite — reads binary uint16 shards
├── eval/
│   ├── __init__.py
│   ├── hellaswag.py          # HellaSwag eval loop
│   └── data/
│       └── hellaswag_val.jsonl
└── configs/
    └── model_config.yaml     # All hyperparameters
```

---

## Architecture (`model/gpt.py`)

### GPTConfig

```python
@dataclass
class GPTConfig:
    vocab_size:     int   = 50304   # 50257 padded to nearest multiple of 64
    context_length: int   = 1024
    d_model:        int   = 768
    n_heads:        int   = 12
    n_layers:       int   = 12
    bias:           bool  = False   # no bias in Linear or LayerNorm
```

**Why vocab_size=32064?**
GPU tensor cores prefer dimensions that are multiples of 64 (or 128). Padding from 32,000 → 32,064 costs 64 × 768 = ~50K extra parameters but improves matmul efficiency. Extra token IDs are never assigned — they're padding for hardware alignment.

---

### CausalSelfAttention

```
Input (B, T, C)
    │
    ├── c_attn: Linear(C, 3C)   → split into Q, K, V  (B, T, C) each
    │       reshape → (B, n_heads, T, head_dim)
    │
    ├── F.scaled_dot_product_attention(Q, K, V, is_causal=True)
    │       Flash Attention — O(N) memory, ~2x faster than naive
    │
    └── c_proj: Linear(C, C)    → output projection
            NANOGPT_SCALE_INIT flag → weight init std scaled by 1/√(2 × n_layers)
```

No dropout. No bias. Flash Attention via `torch.nn.functional.scaled_dot_product_attention`.

---

### MLP

```
Linear(C, 4C) → GELU(approximate='tanh') → Linear(4C, C)
                                                │
                                        NANOGPT_SCALE_INIT flag
```

GELU `approximate='tanh'` matches GPT-2 original and is faster than exact GELU.

---

### Block (Pre-LN Transformer)

```
x = x + Attention(LayerNorm(x))    # pre-norm: LN before attention
x = x + MLP(LayerNorm(x))          # pre-norm: LN before MLP
```

Pre-LN (vs original post-LN in "Attention is All You Need") stabilizes training at scale.

---

### GPT

```
Input token ids (B, T)
    │
    ├── wte: Embedding(vocab_size, d_model)     # token embeddings
    ├── wpe: Embedding(context_length, d_model) # learned positional embeddings
    │
    ├── 12 × Block
    │
    ├── ln_f: LayerNorm(d_model)
    │
    └── lm_head: Linear(d_model, vocab_size, bias=False)
                    ↑
            weight tied to wte.weight           # saves ~24.6M params
```

**Weight tying:** `lm_head.weight = wte.weight`
Input and output embeddings share the same matrix. Tokens that appear in similar contexts get similar embeddings from both directions — empirically improves perplexity and reduces parameters.

---

### Weight Initialization

```
Linear weights:  N(0, 0.02)
                 if NANOGPT_SCALE_INIT: std *= (2 × n_layers)^{-0.5}
Linear biases:   zeros (only exists in non-bias=False layers)
Embeddings:      N(0, 0.02)
```

The scaled init prevents residual stream variance from growing with depth. Each residual add contributes variance — scaling by `1/√(2L)` keeps the total residual variance O(1).

---

### Parameter Count

| Component                        | Parameters                        |
| -------------------------------- | --------------------------------- |
| Token embeddings (wte = lm_head) | 32,064 × 768 = 24,625,152 (24.6M) |
| Position embeddings (wpe)        | 1,024 × 768 = 786,432 (0.8M)      |
| 12 × Attention (c_attn + c_proj) | 12 × (1.77M + 0.59M) = 28.3M      |
| 12 × MLP (c_fc + c_proj)         | 12 × (2.36M + 2.36M) = 56.6M      |
| LayerNorms (bias=False)          | 0.02M                             |
| **Total**                        | **110,365,440 (~110M)**           |

---

## DataLoader (`model/dataloader.py`)

Reads binary uint16 shard files produced by Phase 1's `ShardEncoder`.

```
Shards on disk: shard_train_XXXX.bin, shard_val_XXXX.bin
    │
    ├── np.fromfile(shard, dtype=np.uint16) → torch.long
    ├── current_position advances by B × T × num_processes each step
    └── wraps to next shard when exhausted
```

**Multi-process aware:** Each DDP rank reads a different slice of the same shard (`offset = B × T × rank`), ensuring no two processes see the same tokens.

**Val split:** Reserve last shard as validation. Never seen during training.

---

## Training Loop (`model/train.py`)

### Batch Size

```
total_batch_tokens = 524,288    # 2^19 ≈ 0.5M tokens (matches original)
B (micro batch)    = 64
T (seq length)     = 1024
grad_accum_steps   = 524288 // (64 × 1024) = 8    # single GPU H100
```

### LR Schedule

Cosine decay with linear warmup:

```
Step 0 → 2000:    linear warmup    0 → 6e-4
Step 2000 → 9537: cosine decay     6e-4 → 6e-5
Step > 9537:      constant         6e-5
```

### Optimizer

AdamW with parameter group separation:

- **2D params** (weight matrices, embeddings): weight_decay=0.1
- **1D params** (biases, LayerNorm): weight_decay=0.0
- betas=(0.9, 0.95), eps=1e-8
- `fused=True` on CUDA (faster kernel, fewer memory round-trips)

### Training Step

```python
for micro_step in range(grad_accum_steps):
    with autocast(dtype=bfloat16):
        logits, loss = model(x, y)
    loss = loss / grad_accum_steps    # normalize for mean (not sum)
    loss.backward()

clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
```

Gradient clipping at 1.0 prevents loss spikes from exploding gradients.

### Mixed Precision

`torch.autocast(device_type='cuda', dtype=torch.bfloat16)`

BF16 vs FP16: BF16 has the same exponent range as FP32 — no loss scaling needed. H100 has native BF16 tensor cores. Never use FP16 on H100.

---

## Evaluation

### Validation Loss

- Every 250 steps
- 20 batches × 64 × 1024 tokens = ~1.3M tokens
- `model.eval()` + `torch.no_grad()`

### HellaSwag Accuracy

- Every 250 steps
- 4-choice sentence completion benchmark
- Random baseline = 25%, GPT-2 124M OpenAI = ~29.4%
- Target at 5BT: >30%

### Text Generation Sample

- Every 250 steps
- 4 sequences, max 32 tokens, top-k=50 sampling
- Qualitative sanity check only

---

## Checkpointing

Format: `safetensors` (replaces `.pt`)

**Why safetensors over .pt?**

- `.pt` uses `pickle` — arbitrary code execution on load, security risk
- `safetensors` is fast (zero-copy mmap), safe, and natively supported by HuggingFace Hub

```python
# Saved every 5000 steps
checkpoint = {
    "model":     model.state_dict(),          # safetensors file
    "optimizer": optimizer.state_dict(),       # separate .pt (not on Hub)
    "config":    asdict(model.config),         # metadata JSON
    "step":      step,
    "val_loss":  val_loss,
}
```

Optimizer state is saved locally only (for resume). Only model weights go to HF Hub.

---

## W&B Logging

```python
wandb.log({
    "train/loss":        loss_accum,
    "train/lr":          lr,
    "train/grad_norm":   norm,
    "train/tokens_seen": step * total_batch_tokens,
    "train/tok_per_sec": tokens_per_sec,
    "eval/val_loss":     val_loss,          # every 250 steps
    "eval/hellaswag_acc": acc_norm,         # every 250 steps
}, step=step)
```

Log `tokens_seen` on x-axis (not steps) — makes runs with different batch sizes comparable.

---

## Compute Budget

```
Total tokens:        5,000,000,000
Batch size:          524,288 tokens
Total steps:         ~9,537
H100 throughput:     ~400,000 tok/sec (estimated, BF16 + Flash Attn)
Estimated time:      ~3.5 hours
Cost at $3.01/hr:    ~$10.54
Remaining credits:   ~$4.44 buffer
```

---

## Step-by-Step Build Order

```
1. model/gpt.py          ← Architecture (GPTConfig, CausalSelfAttention, MLP, Block, GPT)
2. tests/test_gpt.py     ← Unit tests: forward pass, loss shape, weight tying, param count
3. model/dataloader.py   ← DataLoaderLite for binary shard files
4. model/eval/hellaswag.py ← HellaSwag eval (port from existing code, adapt tokenizer)
5. model/train.py        ← Full training loop with W&B
6. model/scripts/run_training.sh ← Lightning AI entrypoint
7. Smoke test locally    ← 10 steps on CPU, verify shapes and loss decreases
8. Push to Lightning AI  ← Full 5BT H100 run
```

---

## Known Risks

| Risk                         | Mitigation                                                                                                                                              |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| OOM on H100                  | 110M at BF16: weights=220MB, optimizer=1.32GB, gradients=441MB, activations=13.3GB (Flash Attn) → **~15.3GB total**, well within 80GB (64.7GB headroom) |
| Loss divergence              | Grad clip=1.0 + warmup=2000 steps covers this                                                                                                           |
| Shard exhaustion             | 49M tokens in current shards — need to regenerate at `num_samples=5M+` before training                                                                  |
| HellaSwag tokenizer mismatch | Original used tiktoken GPT-2 tokens — must re-encode HellaSwag with our tokenizer                                                                       |

---

## Pre-Training Checklist

- [x] `model/gpt.py` written and tested
- [ ] `model/dataloader.py` written and tested
- [ ] Shards regenerated at scale (5B tokens worth)
- [ ] HellaSwag data re-encoded with our tokenizer
- [ ] W&B project `llm-playground` created and `wandb login` done
- [ ] `model/train.py` smoke tested locally (10 steps, CPU)
- [ ] Lightning AI studio set up with correct Python env
