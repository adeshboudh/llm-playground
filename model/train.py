"""
train.py — GPT training script
Single GPU:   python train.py
Multi GPU:    torchrun --standalone --nproc_per_node=8 train.py
"""

import os
import math
import time
import yaml
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.gpt import GPT, GPTConfig
from model.dataloader import DataLoaderLite
from model.eval.hellaswag import evaluate_hellaswag
from data.tokenizer.trainer import BPETokenizerTrainer

import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true", help="Fast local CPU smoke test")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------
with open("model/configs/model_config.yaml") as f:
    cfg = yaml.safe_load(f)

mcfg  = cfg["model"]
tcfg  = cfg["training"]
paths = cfg["paths"]
lcfg  = cfg.get("logging", {})

# ---------------------------------------------------------------------------
# DDP setup
# ---------------------------------------------------------------------------
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available(), "DDP requires CUDA"
    init_process_group(backend="nccl")
    ddp_rank       = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device         = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"

device_type = "cuda" if device.startswith("cuda") else "cpu"
if master_process:
    print(f"device: {device}")

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# ---------------------------------------------------------------------------
# Wandb
# ---------------------------------------------------------------------------
lcfg = cfg.get("logging", {})
use_wandb = lcfg.get("wandb_enabled", False) and master_process
if use_wandb:
    wandb.init(
        project=lcfg["wandb_project"],
        name=lcfg["wandb_run_name"],
        config={**mcfg, **tcfg},
    )

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

total_batch_size = tcfg["total_batch_size"]
B                = tcfg["micro_batch_size"]
T                = mcfg["context_length"]
max_steps        = tcfg["max_steps"]
warmup_steps     = tcfg["warmup_steps"]
max_lr           = tcfg["max_lr"]
min_lr           = tcfg["min_lr"]
val_interval     = tcfg["val_interval"]
hella_interval   = tcfg["hella_interval"]
ckpt_interval    = tcfg["ckpt_interval"]
log_dir          = paths["log_dir"]
TOKENIZER_PATH   = paths["tokenizer"]
SHARDS_DIR       = paths["shards_dir"]

if args.smoke:
    B                = 2
    T                = 64
    total_batch_size = 256      # 2 × 64 × 2 steps = tiny
    max_steps        = 5
    val_interval     = 1
    hella_interval   = 999999   # skip hellaswag entirely on smoke
    ckpt_interval    = 999999
    print("⚠️  SMOKE TEST MODE — reduced batch, 5 steps only")

assert total_batch_size % (B * T * ddp_world_size) == 0, \
    "total_batch_size must be divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)

if master_process:
    print(f"total batch size : {total_batch_size:,} tokens")
    print(f"grad accum steps : {grad_accum_steps}")

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
tokenizer = BPETokenizerTrainer.load(TOKENIZER_PATH)
assert tokenizer.vocab_size <= 32064, \
    f"vocab_size mismatch: tokenizer has {tokenizer.vocab_size}, expected ≤32064"

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
train_loader = DataLoaderLite(
    SHARDS_DIR, split="train", B=B, T=T,
    process_rank=ddp_rank, num_processes=ddp_world_size
)
val_loader = DataLoaderLite(
    SHARDS_DIR, split="val", B=B, T=T,
    process_rank=ddp_rank, num_processes=ddp_world_size
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
torch.set_float32_matmul_precision("high")

model = GPT(GPTConfig(**mcfg))
model.to(device)

raw_model = model

use_compile = device_type == "cuda"
if use_compile:
    model = torch.compile(model)

if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])

if master_process:
    total_params = sum(p.numel() for p in raw_model.parameters())
    print(f"parameters: {total_params:,}")

# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------
optimizer = raw_model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=max_lr,
    device_type=device_type,
    verbose=master_process,
)

# ---------------------------------------------------------------------------
# LR scheduler — linear warmup + cosine decay
# ---------------------------------------------------------------------------
def get_lr(step: int) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
if master_process:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "log.txt")
    open(log_file, "w").close()


def log(msg: str, metrics: dict | None = None) -> None:
    if master_process:
        print(msg)
        with open(log_file, "a") as f:
            f.write(msg + "\n")
        if use_wandb and metrics:
            wandb.log(metrics)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
for step in range(max_steps):
    t0 = time.time()
    last_step = step == max_steps - 1

    # ── Validation loss ────────────────────────────────────────────────────
    if step % val_interval == 0 or last_step:
        model.eval()
        val_loader.reset()
        val_loss_accum = 0.0
        val_steps = 20
        with torch.no_grad():
            for _ in range(val_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    _, loss = model(x, y)
                val_loss_accum += loss.detach() / val_steps
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        log(f"{step} val {val_loss_accum.item():.4f}", 
                {"val/loss": val_loss_accum.item(), "step": step})

    # ── Checkpoint ─────────────────────────────────────────────────────────
    if master_process and (step > 0 and step % ckpt_interval == 0 or last_step):
        ckpt_path = os.path.join(log_dir, f"model_{step:05d}.pt")
        torch.save({
            "model":    raw_model.state_dict(),
            "config":   raw_model.config,
            "step":     step,
            "val_loss": val_loss_accum.item(),
            "optimizer": optimizer.state_dict(),
        }, ckpt_path)
        print(f"checkpoint saved → {ckpt_path}")

    # ── HellaSwag eval ─────────────────────────────────────────────────────
    if step % hella_interval == 0 or last_step:
        if master_process:
            model.eval()
            results = evaluate_hellaswag(
                raw_model, tokenizer, device,
                max_examples=200,       # ~2s; set None for full 10,042
            )
            log(f"{step} hella {results['acc_norm']:.4f}", 
                    {"eval/hellaswag_acc_norm": results["acc_norm"], "step": step})

    # ── Training step ──────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y)
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    lr   = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    optimizer.step()

    if device_type == "cuda":
        torch.cuda.synchronize()

    dt  = time.time() - t0
    tps = B * T * grad_accum_steps * ddp_world_size / dt
    log(
        f"{step} train {loss_accum.item():.6f} | lr {lr:.4e} | norm {norm:.4f} | {dt*1000:.0f}ms | {tps:.0f} tok/s",
        {"train/loss": loss_accum.item(), "train/lr": lr,
         "train/grad_norm": norm.item(), "train/tok_per_sec": tps, "step": step}
    )

if use_wandb:
    wandb.finish()

if ddp:
    destroy_process_group()
