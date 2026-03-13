# posttrain/rl/grpo.py
"""
GRPO — Group Relative Policy Optimization
Policy : SFT LoRA model  (only LoRA params update)
Ref    : Frozen copy of SFT model  (KL anchor)
Reward : RM scalar head  (frozen)
Device : CPU / CUDA  (no TPU)
"""
import os, math, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from posttrain.sft.infer_sft   import load_sft_model
from posttrain.reward_model.infer_rm import load_rm, score as rm_score
from posttrain.rl.config import *

os.makedirs(LOG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GRPO | device={device} | G={G} | lr={LR} | kl_coef={KL_COEF}")

# ── Load models ───────────────────────────────────────────────────────────────
policy,  tok, _ = load_sft_model(device=device)
ref_model, _,_ = load_sft_model(device=device)   # frozen copy
rm_model,  rm_tok, _ = load_rm(device=device)

# Freeze reference + RM completely
for p in ref_model.parameters(): p.requires_grad_(False)
for p in rm_model.parameters():  p.requires_grad_(False)
ref_model.eval(); rm_model.eval()

# Only LoRA params in policy are trainable
trainable = [p for p in policy.parameters() if p.requires_grad]
print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
optimizer = torch.optim.AdamW(trainable, lr=LR, betas=(0.9, 0.95))

# ── Prompt dataset ────────────────────────────────────────────────────────────
class PromptDataset(Dataset):
    def __init__(self, path):
        with open(path) as f:
            self.prompts = [l.strip() for l in f if l.strip()]

    def __len__(self): return len(self.prompts)
    def __getitem__(self, idx): return self.prompts[idx]

prompts_ds = PromptDataset(PROMPT_FILE)
loader     = DataLoader(prompts_ds, batch_size=1, shuffle=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def format_prompt(p: str) -> str:
    return f"\n<|user|>\n{p}\n<|assistant|>\n"

def get_logprobs(
    model,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """
    Returns per-token log-probs for the response tokens only.
    input_ids: (1, prompt_len + response_len)
    response_start: index where response begins
    """
    with torch.no_grad() if not model.training else torch.enable_grad():
        logits = model(input_ids=input_ids).logits    # (1, T, V)
    # logits at position t predicts token t+1
    shift_logits = logits[0, response_start-1 : -1, :]   # (resp_len, V)
    shift_labels = input_ids[0, response_start:]          # (resp_len,)
    logp = F.log_softmax(shift_logits, dim=-1)
    return logp.gather(1, shift_labels.unsqueeze(1)).squeeze(1)  # (resp_len,)


def generate_response(prompt_ids: torch.Tensor) -> torch.Tensor:
    """Sample one response, returns full input_ids (prompt + response)."""
    with torch.no_grad():
        out = policy.generate(
            prompt_ids,
            max_new_tokens = MAX_NEW,
            temperature    = TEMPERATURE,
            top_k          = TOP_K,
            do_sample      = True,
            pad_token_id   = tok.eos_token_id,
            eos_token_id   = tok.eos_token_id,
        )
    return out   # (1, prompt_len + response_len)


def grpo_step(prompt: str) -> dict:
    """One GRPO update for a single prompt with G responses."""
    prompt_text = format_prompt(prompt)
    prompt_ids  = tok.encode(prompt_text, return_tensors="pt").to(device)
    prompt_len  = prompt_ids.shape[1]

    # ── 1. Sample G responses from current policy ─────────────────────────────
    responses, rewards = [], []
    policy.eval()   # no dropout during sampling
    with torch.no_grad():
        for _ in range(G):
            full_ids = generate_response(prompt_ids)     # (1, T)
            resp_text = tok.decode(
                full_ids[0, prompt_len:], skip_special_tokens=True
            ).strip()
            r = rm_score(prompt, resp_text, rm_model, rm_tok, device)
            responses.append(full_ids)
            rewards.append(r)

    rewards_t = torch.tensor(rewards, device=device)

    # ── 2. Group-relative advantages ─────────────────────────────────────────
    mean_r = rewards_t.mean()
    std_r  = rewards_t.std() + 1e-8
    advantages = (rewards_t - mean_r) / std_r   # (G,)

    # ── 3. Compute GRPO loss over all G responses ─────────────────────────────
    policy.train()
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for i, (full_ids, adv) in enumerate(zip(responses, advantages)):
        resp_len = full_ids.shape[1] - prompt_len
        if resp_len == 0:
            continue

        # Log probs from CURRENT policy (with grad)
        logp_policy = get_logprobs(policy,    full_ids, prompt_len)  # (R,)

        # Log probs from OLD policy (frozen, no grad)
        with torch.no_grad():
            logp_old = get_logprobs(ref_model, full_ids, prompt_len) # (R,)

        # Per-token importance ratio
        ratio = torch.exp(logp_policy - logp_old)  # (R,)

        # Clipped surrogate
        clipped = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        pg_loss = -adv * torch.min(ratio, clipped)           # (R,)

        # KL penalty  (per token, averaged over response)
        with torch.no_grad():
            logp_ref = get_logprobs(ref_model, full_ids, prompt_len)
        kl = (logp_policy - logp_ref).mean()

        loss_i = pg_loss.mean() + KL_COEF * kl
        total_loss = total_loss + loss_i / G

    # ── 4. Backprop ───────────────────────────────────────────────────────────
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()

    return {
        "loss":     total_loss.item(),
        "reward":   mean_r.item(),
        "rewards":  [round(r, 3) for r in rewards],
        "adv_std":  std_r.item(),
    }


# ── Training loop ─────────────────────────────────────────────────────────────
log_path = os.path.join(LOG_DIR, "grpo_log.txt")
with open(log_path, "w") as f:
    f.write("step,loss,mean_reward,adv_std\n")

step = 0
for epoch in range(math.ceil(MAX_STEPS / len(prompts_ds))):
    for (prompt,) in loader:
        if step >= MAX_STEPS:
            break
        t0 = time.perf_counter()
        stats = grpo_step(prompt[0] if isinstance(prompt, list) else prompt)
        dt = time.perf_counter() - t0

        print(
            f"step {step:4d} | loss: {stats['loss']:+.4f} | "
            f"mean_r: {stats['reward']:+.4f} | "
            f"rewards: {stats['rewards']} | "
            f"dt: {dt:.1f}s"
        )
        with open(log_path, "a") as f:
            f.write(f"{step},{stats['loss']:.6f},{stats['reward']:.6f},{stats['adv_std']:.6f}\n")

        if step > 0 and step % SAVE_EVERY == 0:
            ckpt_path = f"checkpoints/rl/grpo_step_{step:04d}.pt"
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            lora_state = {k: v.cpu() for k, v in policy.state_dict().items()
                          if "lora_" in k}
            torch.save({"lora_state_dict": lora_state, "step": step}, ckpt_path)
            print(f"  [CKPT] → {ckpt_path}")

        step += 1

print("GRPO training complete.")
