# posttrain/sft/infer_sft.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
from posttrain.sft.config import (
    CHECKPOINT_PATH, HUMAN_HEADER, ASST_HEADER,
    LORA_R, LORA_ALPHA, LORA_DROPOUT, LORA_TARGETS, BASE_MODEL_ID,
)

def load_sft_model(
    adapter_path: str = CHECKPOINT_PATH,
    device: str | None = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    base = GPT2LMHeadModel.from_pretrained(BASE_MODEL_ID)
    cfg  = LoraConfig(
        task_type      = TaskType.CAUSAL_LM,
        r              = LORA_R,
        lora_alpha     = LORA_ALPHA,
        target_modules = LORA_TARGETS,
        lora_dropout   = LORA_DROPOUT,
        bias           = "none",
        inference_mode = True,
    )
    model = get_peft_model(base, cfg)
    ckpt  = torch.load(adapter_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    model.to(device).eval()

    tok           = GPT2Tokenizer.from_pretrained(BASE_MODEL_ID)
    tok.pad_token = tok.eos_token
    return model, tok, device


def generate(
    prompt: str,
    model,
    tokenizer,
    device: str,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_k: int = 40,
    repetition_penalty: float = 1.2,
) -> str:
    full = f"{HUMAN_HEADER}{prompt}{ASST_HEADER}"
    ids  = tokenizer.encode(full, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens      = max_new_tokens,
            temperature         = temperature,
            top_k               = top_k,
            repetition_penalty  = repetition_penalty,
            do_sample           = True,
            pad_token_id        = tokenizer.eos_token_id,
            eos_token_id        = tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][ids.shape[1]:], skip_special_tokens=True
    ).strip()


if __name__ == "__main__":
    model, tok, device = load_sft_model()
    tests = [
        "What is the capital of France?",
        "Explain machine learning in simple terms.",
        "Write a Python function to reverse a string.",
    ]
    for p in tests:
        print(f"\n>>> {p}")
        print(generate(p, model, tok, device))
