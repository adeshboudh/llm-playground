# posttrain/rl/config.py

# ── GRPO hyperparams ──────────────────────────────────────────────────────────
G           = 4       # responses per prompt (group size)
CLIP_EPS    = 0.2     # PPO clip ratio
KL_COEF     = 0.04    # KL penalty coefficient (β)
MAX_NEW     = 80      # max tokens per response
TEMPERATURE = 0.9     # sampling temperature — higher = more diverse
TOP_K       = 50

# ── Training ──────────────────────────────────────────────────────────────────
LR          = 5e-6    # very small — LoRA on top of frozen GPT-2
MAX_STEPS   = 200     # ~800 policy updates total (200 prompts × G=4)
SAVE_EVERY  = 50
LOG_DIR     = "log/grpo"
PROMPT_FILE = "posttrain/rl/prompts.txt"
