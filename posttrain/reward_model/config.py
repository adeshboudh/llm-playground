# posttrain/reward_model/config.py
import os

CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "../../checkpoints/reward_model/rm_final.pt"
)

BASE_MODEL_ID = "gpt2"   # must match train_rm.py
HIDDEN_SIZE   = 768      # GPT-2 n_embd

# Data format — must match prepare_rm_data.py exactly
# hh-rlhf format: "\n\nHuman: ...\n\nAssistant: ..."
HUMAN_PREFIX = "\n\nHuman:"
ASST_PREFIX  = "\n\nAssistant:"
