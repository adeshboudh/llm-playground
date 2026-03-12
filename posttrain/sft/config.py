# posttrain/sft/config.py
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "../../checkpoints/sft/adapter_sft_v1.pt"
)

# ── Chat template — MUST match tokenize_ultrachat.py exactly ─────────────────
# tokenize_ultrachat.py uses "\n<|user|>\n" and "\n<|assistant|>\n"
HUMAN_HEADER = "\n<|user|>\n"
ASST_HEADER  = "\n<|assistant|>\n"

# ── LoRA config — MUST match train_sft_lora.py exactly ───────────────────────
LORA_R        = 16
LORA_ALPHA    = 32
LORA_DROPOUT  = 0.05
LORA_TARGETS  = ["c_attn", "c_proj"]
BASE_MODEL_ID = "gpt2"
