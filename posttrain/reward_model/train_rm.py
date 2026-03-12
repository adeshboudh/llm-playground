"""
Reward Model — TPU v5e-8 SPMD
Trains GPT-2 + scalar head to predict "chosen > rejected"
"""
import os, math, time, glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import torch_xla.distributed.spmd as xs

from transformers import GPT2Model, GPT2Config
import tiktoken

# ── SPMD first ────────────────────────────────────────────────────────────────
xr.use_spmd()

# ── Config ────────────────────────────────────────────────────────────────────
SHARD_DIR   = "/kaggle/working/rm_shards"
LOG_DIR     = "/kaggle/working/log/rm"
MAX_LR      = 1e-4
MIN_LR      = 1e-5
WARMUP_STEPS= 50
MAX_STEPS   = 1000
WEIGHT_DECAY= 0.01
GRAD_CLIP   = 1.0
SAVE_EVERY  = 200
GRAD_ACCUM  = 4
B           = 8
T           = 1024

os.makedirs(LOG_DIR, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device    = torch_xla.device()
num_cores = xr.global_runtime_device_count()
mesh      = xs.Mesh(list(range(num_cores)), (num_cores, 1), ("data", "model"))

xm.master_print(f"RM training | Cores: {num_cores} | Global batch: {B * num_cores * GRAD_ACCUM * T:,}")

# ── BUG 1 FIX: define load_shard ─────────────────────────────────────────────
def load_shard(path):
    data = np.load(path)
    ids    = torch.from_numpy(data["ids"].astype(np.int64))
    labels = torch.from_numpy(data["labels"].astype(np.int64))
    return ids, labels

# ── Dataset ───────────────────────────────────────────────────────────────────
class RMDataset(Dataset):
    def __init__(self, split):
        shards = sorted(glob.glob(os.path.join(SHARD_DIR, f"{split}_*.npz")))
        assert len(shards) > 0, f"No shards for {split}"
        all_ids, all_labels = zip(*[load_shard(p) for p in shards])
        self.ids    = torch.cat(all_ids)
        self.labels = torch.cat(all_labels)
        self.n      = (len(self.ids) - 1) // T
        xm.master_print(f"[{split}] {len(shards)} shards | {self.n:,} windows")

    def __len__(self): return self.n

    def __getitem__(self, idx):
        pos = idx * T
        return (self.ids[pos : pos + T],
                self.labels[pos : pos + T])

train_ds     = RMDataset("rm_train")
train_loader = DataLoader(train_ds, batch_size=B, shuffle=True,
                          num_workers=4, drop_last=True)

# ── BUG 2 & 5 FIX: proper model class instead of nn.Sequential ───────────────
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained("gpt2")  # returns tensors directly
        hidden_size      = self.transformer.config.n_embd     # 768
        self.reward_head = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, input_ids):
        # transformer returns BaseModelOutput; .last_hidden_state is (B, T, H)
        hidden = self.transformer(input_ids=input_ids).last_hidden_state
        # mean-pool over sequence → (B, H) → (B, 1) → (B,)
        reward = self.reward_head(hidden.mean(dim=1)).squeeze(-1)
        return reward

xm.master_print("Loading GPT-2 + RM head...")
model = RewardModel().to(device)

# Shard model weights across mesh
for param in model.parameters():
    xs.mark_sharding(param, mesh, (None,) * param.dim())

param_count = sum(p.numel() for p in model.parameters())
xm.master_print(f"Parameters: {param_count:,} ({param_count/1e6:.0f}M)")

optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR,
                              betas=(0.9, 0.95), weight_decay=WEIGHT_DECAY)

# ── LR schedule ───────────────────────────────────────────────────────────────
def get_lr(step):
    if step < WARMUP_STEPS:
        return MAX_LR * (step + 1) / WARMUP_STEPS
    if step >= MAX_STEPS:
        return MIN_LR
    ratio = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
    return MIN_LR + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (MAX_LR - MIN_LR)

# ── Loss ──────────────────────────────────────────────────────────────────────
def preference_loss(rewards, labels):
    # BUG 3 FIX: binarize the mean label instead of using a soft float target
    binary_labels = (labels.float().mean(dim=1) > 0.5).float()
    return F.binary_cross_entropy_with_logits(rewards, binary_labels)

# ── Training loop ─────────────────────────────────────────────────────────────
log_file   = os.path.join(LOG_DIR, "rm_log.txt")
train_iter = iter(train_loader)
step       = 0

with open(log_file, "w") as f:
    pass

while step < MAX_STEPS:
    t0 = time.perf_counter()
    loss_accum = torch.zeros(1, device=device)
    optimizer.zero_grad(set_to_none=True)

    for _ in range(GRAD_ACCUM):
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x = x.to(device)
        y = y.to(device)
        xs.mark_sharding(x, mesh, ("data", None))
        xs.mark_sharding(y, mesh, ("data", None))

        # BUG 6: autocast on XLA — safe in torch_xla >= 2.x, remove if you hit issues
        with torch.autocast("xla", dtype=torch.bfloat16):
            rewards = model(x)                          # (B,)
            loss    = preference_loss(rewards, y) / GRAD_ACCUM

        loss.backward()
        loss_accum += loss.detach()

    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    xm.optimizer_step(optimizer)
    torch_xla.sync()

    dt       = time.perf_counter() - t0
    tps      = (GRAD_ACCUM * B * num_cores * T) / dt
    loss_val = loss_accum.item()

    xm.master_print(f"step {step:4d} | loss: {loss_val:.4f} | "
                    f"lr: {lr:.2e} | dt: {dt*1000:.0f}ms | tok/sec: {tps:.0f}")

    if step > 0 and step % SAVE_EVERY == 0:
        path = os.path.join(LOG_DIR, f"rm_{step:05d}")
        os.makedirs(path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(path, "rm.pt"))
        xm.master_print(f"  [CKPT] saved → {path}")

    step += 1

# Final save
torch.save(model.state_dict(), os.path.join(LOG_DIR, "rm_final.pt"))
xm.master_print("Reward model training complete.")
