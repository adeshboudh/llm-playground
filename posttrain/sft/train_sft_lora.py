"""
LoRA SFT — TPU v5e-8, SPMD data parallel
Single process, 8 cores, BF16 via XLA autocast
"""
import os, math, time, glob, json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

# ── SPMD must be first — before any device creation ───────────────────────────
xr.use_spmd()

from transformers import GPT2LMHeadModel
from peft import get_peft_model, LoraConfig, TaskType
import tiktoken

# ── Config ────────────────────────────────────────────────────────────────────
SHARD_DIR    = "/kaggle/working/ultrachat_shards"
LOG_DIR      = "/kaggle/working/log/sft_lora"
MAX_LR       = 3e-4
MIN_LR       = 3e-5
WARMUP_STEPS = 50
MAX_STEPS    = 1000
WEIGHT_DECAY = 0.01
GRAD_CLIP    = 1.0
EVAL_EVERY   = 100
SAVE_EVERY   = 100
GRAD_ACCUM   = 4
B            = 8      # per-core batch size  → global = 8×8 = 64
T            = 1024
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05

os.makedirs(LOG_DIR, exist_ok=True)

# ── Device + SPMD mesh ────────────────────────────────────────────────────────
device     = torch_xla.device()
num_cores  = xr.global_runtime_device_count()
mesh       = xs.Mesh(list(range(num_cores)), (num_cores, 1), ("data", "model"))

xm.master_print(f"Device: {device} | Cores: {num_cores}")
xm.master_print(f"Global batch: {B * num_cores * GRAD_ACCUM * T:,} tokens/step")

# ── Dataset ───────────────────────────────────────────────────────────────────
def load_shard(path):
    data   = np.load(path)
    id_key = "ids" if "ids" in data else "input_ids"
    return (torch.tensor(data[id_key],    dtype=torch.long),
            torch.tensor(data["labels"],  dtype=torch.long))

class SFTShardDataset(Dataset):
    def __init__(self, split):
        shards = sorted(glob.glob(os.path.join(SHARD_DIR, f"{split}_*.npz")))
        assert len(shards) > 0, f"No shards for {split}"
        all_ids, all_labels = zip(*[load_shard(p) for p in shards])
        self.ids    = torch.cat(all_ids)
        self.labels = torch.cat(all_labels)
        self.n      = (len(self.ids) - 1) // T
        xm.master_print(f"[{split}] {len(shards)} shards | {self.n:,} windows")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        pos = idx * T
        return (self.ids   [pos     : pos + T],
                self.labels[pos + 1 : pos + T + 1])

# ── Model + LoRA ──────────────────────────────────────────────────────────────
xm.master_print("Loading GPT-2 + LoRA...")
base = GPT2LMHeadModel.from_pretrained("gpt2")
lora_cfg = LoraConfig(
    task_type    = TaskType.CAUSAL_LM,
    r            = LORA_R,
    lora_alpha   = LORA_ALPHA,
    target_modules=["c_attn", "c_proj"],
    lora_dropout = LORA_DROPOUT,
    bias         = "none",
    inference_mode=False,
)
model = get_peft_model(base, lora_cfg).to(device)

# Shard model params across data axis (replicates weights on all 8 cores)
for param in model.parameters():
    xs.mark_sharding(param, mesh, (None,) * param.dim())

model.print_trainable_parameters()

# ── Optimizer ─────────────────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=MAX_LR, betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY,
)

# ── Data loaders ──────────────────────────────────────────────────────────────
train_ds = SFTShardDataset("train")
val_ds   = SFTShardDataset("val")
train_loader = DataLoader(train_ds, batch_size=B, shuffle=True,
                          num_workers=4, drop_last=True)
val_loader   = DataLoader(val_ds,   batch_size=B, shuffle=False,
                          num_workers=4, drop_last=True)

# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (MAX_LR - MIN_LR)

# ── Eval ──────────────────────────────────────────────────────────────────────
def evaluate():
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device); y = y.to(device)
            xs.mark_sharding(x, mesh, ("data", None))
            xs.mark_sharding(y, mesh, ("data", None))
            with torch.autocast("xla", dtype=torch.bfloat16):
                out = model(input_ids=x, labels=y)
            total += out.loss
            n     += 1
            torch_xla.sync()
            if n >= 20: break
    model.train()
    return (total / n).item()

# ── Training loop ─────────────────────────────────────────────────────────────
log_file   = os.path.join(LOG_DIR, "lora_log.txt")
train_iter = iter(train_loader)
step       = 0

with open(log_file, "w") as f:
    pass

while step < MAX_STEPS:
    t0         = time.perf_counter()
    loss_accum = torch.zeros(1, device=device)
    optimizer.zero_grad(set_to_none=True)

    for _ in range(GRAD_ACCUM):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device); y = y.to(device)

        # Shard batch across 8 cores
        xs.mark_sharding(x, mesh, ("data", None))
        xs.mark_sharding(y, mesh, ("data", None))

        with torch.autocast("xla", dtype=torch.bfloat16):
            out  = model(input_ids=x, labels=y)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        loss_accum += loss.detach()

    # Grad clip + step
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    xm.optimizer_step(optimizer)
    torch_xla.sync()          # ← single sync per step, not per micro-step

    dt  = time.perf_counter() - t0
    tps = (GRAD_ACCUM * B * num_cores * T) / dt

    # .item() AFTER sync — safe, no extra compilation
    loss_val = loss_accum.item()
    xm.master_print(f"step {step:4d} | loss: {loss_val:.4f} | "
                    f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/sec: {tps:.0f}")
    with open(log_file, "a") as f:
        f.write(f"{step} train {loss_val:.6f}\n")

    # Eval — skip step 0 to avoid back-to-back compilations
    if step > 0 and step % EVAL_EVERY == 0:
        val_loss = evaluate()
        xm.master_print(f"  [EVAL] step={step} | val_loss={val_loss:.4f}")
        with open(log_file, "a") as f:
            f.write(f"{step} val {val_loss:.6f}\n")

    # Save
    if step > 0 and step % SAVE_EVERY == 0:
        path = os.path.join(LOG_DIR, f"sft_lora_{step:05d}")
        os.makedirs(path, exist_ok=True)
    
        # Pull LoRA weights to CPU before saving — XLA tensors can't be serialized directly
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()
                     if "lora_" in k}   # save only LoRA delta weights
        torch.save({
            "lora_state_dict": cpu_state,
            "step": step,
            "lora_config": {
                "r": LORA_R, "lora_alpha": LORA_ALPHA,
                "target_modules": ["c_attn", "c_proj"],
                "lora_dropout": LORA_DROPOUT,
            }
        }, os.path.join(path, "adapter.pt"))
        xm.master_print(f"  [CKPT] saved → {path}/adapter.pt")


    step += 1

# Final save
final_path = os.path.join(LOG_DIR, "sft_lora_final")
os.makedirs(final_path, exist_ok=True)
cpu_state = {k: v.cpu() for k, v in model.state_dict().items()
             if "lora_" in k}
torch.save({
    "lora_state_dict": cpu_state,
    "step": MAX_STEPS,
    "lora_config": {
        "r": LORA_R, "lora_alpha": LORA_ALPHA,
        "target_modules": ["c_attn", "c_proj"],
        "lora_dropout": LORA_DROPOUT,
    }
}, os.path.join(final_path, "adapter.pt"))
xm.master_print(f"Final model saved → {final_path}/adapter.pt")
xm.master_print("Done.")
