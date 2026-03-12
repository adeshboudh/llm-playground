# posttrain/reward_model/infer_rm.py
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from posttrain.reward_model.config import CHECKPOINT_PATH, BASE_MODEL_ID, HIDDEN_SIZE


# ── Model — must match train_rm.py exactly ────────────────────────────────────
class RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained(BASE_MODEL_ID)
        self.reward_head = nn.Linear(HIDDEN_SIZE, 1, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.transformer(input_ids=input_ids).last_hidden_state
        return self.reward_head(hidden.mean(dim=1)).squeeze(-1)   # (B,)


def load_rm(
    rm_path: str = CHECKPOINT_PATH,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RewardModel()
    state = torch.load(rm_path, map_location="cpu", weights_only=False)

    # TPU saves keys as "transformer.*" and "reward_head.*" — load directly
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    tok           = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tok.pad_token = tok.eos_token
    return model, tok, device


def score(
    prompt: str,
    response: str,
    model: RewardModel,
    tokenizer: GPT2Tokenizer,
    device: str,
    max_length: int = 1024,
) -> float:
    text = f"{prompt}\n\nAssistant: {response}"
    ids  = tokenizer.encode(
        text, return_tensors="pt",
        max_length=max_length, truncation=True
    ).to(device)
    with torch.no_grad():
        return model(ids).item()


if __name__ == "__main__":
    rm, tok, device = load_rm()

    # Sanity check: good response must score higher than bad
    prompt = "What is the capital of France?"
    tests  = [
        ("Paris is the capital of France.", "good"),
        ("I think it might be London or Berlin.", "bad"),
        ("The capital is Paris, a city known for the Eiffel Tower.", "good+detail"),
    ]
    print(f"Prompt: {prompt}\n")
    scores = []
    for resp, label in tests:
        s = score(prompt, resp, rm, tok, device)
        scores.append(s)
        print(f"  [{label:12s}]  score={s:+.4f}  |  {resp}")

    print(f"\nGood > Bad: {scores[0] > scores[1]}")   # must be True
