"""
GRPO on Kaggle 2×T4 GPU
Policy  : GPT-2 + LoRA (trainable)
Ref     : GPT-2 + LoRA (frozen copy, KL anchor)
Reward  : GPT-2 + scalar head (frozen)
"""
import os, math, time, torch, torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Model
from peft import get_peft_model, LoraConfig, TaskType
import copy

# ── DDP setup ─────────────────────────────────────────────────────────────────
ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    dist.init_process_group("nccl")
    rank       = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device     = f"cuda:{local_rank}"
    torch.cuda.set_device(device)
    is_main    = rank == 0
else:
    rank = local_rank = 0
    world_size = 1
    is_main    = True
    device     = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42 + rank)
if is_main:
    print(f"GRPO | device={device} | world_size={world_size}")

# ── Config ────────────────────────────────────────────────────────────────────
ADAPTER_PATH = "/kaggle/input/models/adeshboudh/adapter-sft-v1/pytorch/default/1/adapter.pt"  # upload adapter.pt as dataset
RM_PATH      = "/kaggle/input/models/adeshboudh/adapter-sft-v1/pytorch/default/1/rm_final.pt"
LOG_DIR      = "/kaggle/working/log/grpo"
os.makedirs(LOG_DIR, exist_ok=True)

MIN_STD  = 0.1    
G           = 4        # responses per prompt
CLIP_EPS    = 0.2
KL_COEF     = 0.04
MAX_NEW     = 80
TEMPERATURE = 0.9
TOP_K       = 50
LR          = 5e-6
MAX_STEPS   = 300
SAVE_EVERY  = 50
LORA_R      = 16
LORA_ALPHA  = 32

# ── Tokenizer ─────────────────────────────────────────────────────────────────
tok               = GPT2Tokenizer.from_pretrained("gpt2")
tok.pad_token     = tok.eos_token

# ── Model loaders ─────────────────────────────────────────────────────────────
def load_policy(adapter_path, trainable=True):
    base = GPT2LMHeadModel.from_pretrained("gpt2")
    cfg  = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = ["c_attn", "c_proj"],
        lora_dropout   = 0.05,
        bias           = "none",
        inference_mode = not trainable,   # ← KEY FIX
    )
    model = get_peft_model(base, cfg)
    ckpt  = torch.load(adapter_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    model.to(device)
    if not trainable:
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    return model

class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained("gpt2")
        self.reward_head = nn.Linear(768, 1, bias=False)
    def forward(self, input_ids):
        h = self.transformer(input_ids=input_ids).last_hidden_state
        return self.reward_head(h.mean(dim=1)).squeeze(-1)

def load_rm(rm_path):
    model = RewardModel()
    state = torch.load(rm_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

# ── Load all three models ─────────────────────────────────────────────────────
if is_main: print("Loading policy...")
policy    = load_policy(ADAPTER_PATH, trainable=True)

if is_main: print("Loading reference (frozen)...")
ref_model = load_policy(ADAPTER_PATH, trainable=False)

if is_main: print("Loading reward model...")
rm_model  = load_rm(RM_PATH)

trainable_params = [p for p in policy.parameters() if p.requires_grad]
if is_main:
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")
    assert len(trainable_params) > 0, "No trainable params! Check LoRA config."

optimizer = torch.optim.AdamW(trainable_params, lr=LR, betas=(0.9, 0.95))

# Wrap policy in DDP
if ddp:
    policy = DDP(policy, device_ids=[local_rank])

raw_policy = policy.module if ddp else policy

# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPTS = [
    "What is the capital of France?",
    "Explain what machine learning is in simple terms.",
    "Write a Python function to reverse a string.",
    "What causes rain?",
    "How does a computer processor work?",
    "What is the difference between a list and a tuple in Python?",
    "Explain the concept of gravity.",
    "What is the French Revolution?",
    "How do you make pasta?",
    "What is photosynthesis?",
    "Describe the water cycle.",
    "What is recursion in programming?",
    "How does the internet work?",
    "What is inflation?",
    "Write a function to check if a number is prime.",
    "What is DNA?",
    "Explain object-oriented programming.",
    "What are the causes of World War 1?",
    "How do vaccines work?",
    "What is the speed of light?",
]

class PromptDataset(Dataset):
    def __init__(self): pass
    def __len__(self): return len(PROMPTS)
    def __getitem__(self, idx): return PROMPTS[idx]

sampler = DistributedSampler(PromptDataset(), num_replicas=world_size,
                              rank=rank, shuffle=True) if ddp else None
loader  = DataLoader(PromptDataset(), batch_size=1,
                     sampler=sampler, shuffle=(sampler is None))

# ── GRPO helpers ──────────────────────────────────────────────────────────────
def format_prompt(p): return f"\n<|user|>\n{p}\n<|assistant|>\n"

def get_logprobs(model, input_ids, response_start):
    logits      = model(input_ids=input_ids).logits             # (1, T, V)
    shift_logits= logits[0, response_start-1:-1, :]             # (R, V)
    shift_labels= input_ids[0, response_start:]                 # (R,)
    logp        = F.log_softmax(shift_logits, dim=-1)
    return logp.gather(1, shift_labels.unsqueeze(1)).squeeze(1) # (R,)

@torch.no_grad()
def generate_response(prompt_ids):
    return raw_policy.generate(
        prompt_ids,
        max_new_tokens = MAX_NEW,
        temperature    = TEMPERATURE,
        top_k          = TOP_K,
        do_sample      = True,
        pad_token_id   = tok.eos_token_id,
        eos_token_id   = tok.eos_token_id,
    )

@torch.no_grad()
def rm_score(prompt, response, device):
    text = f"\n\nHuman: {prompt}\n\nAssistant: {response}"
    ids  = tok.encode(text, return_tensors="pt",
                      max_length=1024, truncation=True).to(device)
    with torch.no_grad():
        return rm_model(ids).item()

# ── GRPO step ─────────────────────────────────────────────────────────────────
def grpo_step(prompt):
    prompt_ids = tok.encode(format_prompt(prompt),
                            return_tensors="pt").to(device)
    prompt_len = prompt_ids.shape[1]

    # 1. Sample G responses + score them
    raw_policy.eval()
    full_ids_list, rewards = [], []
    for _ in range(G):
        full_ids  = generate_response(prompt_ids)
        resp_text = tok.decode(full_ids[0, prompt_len:],
                               skip_special_tokens=True).strip()
        r = rm_score(prompt, resp_text, device)
        full_ids_list.append(full_ids)
        rewards.append(r)

    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)

    # 2. Group-relative advantages — NO CLAMP
    mean_r = rewards_t.mean()
    std_r  = rewards_t.std() + 1e-8

    # Skip degenerate groups (RM gave same score to everything)
    if std_r < MIN_STD:
        return {"loss": 0.0, "mean_r": mean_r.item(),
                "rewards": rewards, "skipped": True}

    advs = (rewards_t - mean_r) / std_r   # z-score: always ~[-2, +2]

    # 3. GRPO loss
    raw_policy.train()
    optimizer.zero_grad()
    total_loss = torch.zeros(1, device=device)

    for full_ids, adv in zip(full_ids_list, advs):
        resp_len = full_ids.shape[1] - prompt_len
        if resp_len == 0:
            continue

        logp_new = get_logprobs(raw_policy, full_ids, prompt_len)

        with torch.no_grad():
            logp_ref = get_logprobs(ref_model, full_ids, prompt_len)

        ratio   = torch.exp(logp_new - logp_ref)
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        pg_loss = -adv * torch.min(ratio, clipped)

        # KL: clamp to ≥0 so it's always a penalty, never a reward
        kl = torch.clamp((logp_new - logp_ref).mean(), min=0.0)

        loss_i     = pg_loss.mean() + KL_COEF * kl
        total_loss = total_loss + loss_i / G

    # 4. Backprop
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
    optimizer.step()

    return {
        "loss":    total_loss.item(),
        "mean_r":  mean_r.item(),
        "rewards": [round(r, 3) for r in rewards],
        "adv_std": std_r.item(),
        "skipped": False,
    }


# ── Training loop ─────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, "grpo_log.txt")
if is_main:
    with open(log_path, "w") as f:
        f.write("step,loss,mean_reward\n")

step = 0
for epoch in range(math.ceil(MAX_STEPS / len(PROMPTS))):
    if sampler: sampler.set_epoch(epoch)
    for (prompt,) in loader:
        if step >= MAX_STEPS: break
        t0    = time.perf_counter()
        stats = grpo_step(prompt[0] if isinstance(prompt, list) else prompt)
        dt    = time.perf_counter() - t0

        if is_main:
            print(f"step {step:4d} | loss: {stats['loss']:+.5f} | "
                  f"mean_r: {stats['mean_r']:+.4f} | adv_std: {stats['adv_std']:.3f} | "
                  f"rewards: {stats['rewards']}")
            with open(log_path, "a") as f:
                f.write(f"{step},{stats['loss']:.6f},{stats['mean_r']:.6f}\n")

        if is_main and step > 0 and step % SAVE_EVERY == 0:
            path = os.path.join(LOG_DIR, f"grpo_{step:04d}.pt")
            lora = {k: v.cpu() for k, v in raw_policy.state_dict().items()
                    if "lora_" in k}
            torch.save({"lora_state_dict": lora, "step": step}, path)
            print(f"  [CKPT] → {path}")
        step += 1

if is_main:
    path = os.path.join(LOG_DIR, "grpo_final.pt")
    lora = {k: v.cpu() for k, v in raw_policy.state_dict().items()
            if "lora_" in k}
    torch.save({"lora_state_dict": lora, "step": step}, path)
    print(f"GRPO done → {path}")

if ddp:
    dist.destroy_process_group()
