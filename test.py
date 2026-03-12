from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

base      = GPT2LMHeadModel.from_pretrained("gpt2")
lora_cfg  = LoraConfig(task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32,
                       target_modules=["c_attn","c_proj"],
                       lora_dropout=0.05, bias="none", inference_mode=True)
model = get_peft_model(base, lora_cfg)
ckpt  = torch.load("checkpoints/sft/adapter.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["lora_state_dict"], strict=False)
model.eval()

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

def chat(prompt, model, tokenizer):
    # Exact match to training shards — User:/Assistant: format
    full_prompt = f"User: {prompt}\nAssistant:"
    ids = tokenizer.encode(full_prompt, return_tensors="pt")
    
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=100,
            temperature=0.7,
            top_k=40,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

print(chat("What is the capital of France?", model, tokenizer))
print(chat("Explain machine learning in simple terms", model, tokenizer))
print(chat("Write a Python function to reverse a string", model, tokenizer))






# import torch, tiktoken
# from model.gpt import GPT, GPTConfig

# ckpt  = torch.load("log/sft/sft_00799.pt", map_location="cpu", weights_only=False)
# state = {k.replace("_orig_mod.", ""): v for k, v in ckpt["model"].items()}
# model = GPT(GPTConfig(vocab_size=50304, context_length=1024,
#                       d_model=768, n_heads=12, n_layers=12, bias=True))
# model.load_state_dict(state)
# model.eval()


# keys = list(ckpt["model"].keys())
# print("First 5 keys:", keys[:5])
# print("Total keys:", len(keys))
# print("Step:", ckpt.get("step"))

# enc    = tiktoken.get_encoding("gpt2")
# EOS_ID = 50256

# def generate(prompt, label):
#     ids = enc.encode(prompt)
#     x   = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
#     out = []
#     with torch.no_grad():
#         for _ in range(80):
#             logits, _ = model(x)
#             logits    = logits[0, -1, :] / 0.8
#             v, _      = torch.topk(logits, 20)
#             logits[logits < v[-1]] = -float("Inf")
#             next_id = torch.multinomial(torch.softmax(logits, -1), 1)
#             if next_id.item() == EOS_ID:
#                 break
#             out.append(next_id.item())
#             x = torch.cat([x, next_id.view(1,1)], dim=1)
#     print(f"\n[{label}]")
#     print(repr(enc.decode(out)))

# # Format A — what my tokenize script used
# generate("\n<human>\nWhat is the capital of France?\n<assistant>\n",
#          "format_A: <human>/<assistant>")

# # Format B — what local shards show
# generate("User: What is the capital of France?\nAssistant:",
#          "format_B: User:/Assistant:")

# # Format C — bare pretrain style
# generate("The capital of France is",
#          "format_C: pretrain style")

# import torch
# import tiktoken
# from model.gpt import GPT, GPTConfig

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {device}")

# # ── Load checkpoint ─────────────────────────────────────────────
# ckpt_path = "log/sft/sft_01500.pt"
# ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
# print(f"Loaded checkpoint from step: {ckpt.get('step', 'unknown')}")

# # ── Rebuild model (ensure this matches training config) ─────────
# config = GPTConfig(
#     vocab_size=50304,
#     context_length=1024,
#     d_model=768,
#     n_heads=12,
#     n_layers=12,
#     bias=True,
# )

# model = GPT(config).to(device)

# # Clean and load state dict
# state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
# state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

# # Strict=False can help if there's a tiny mismatch, but it should ideally be True
# model.load_state_dict(state_dict)
# model.eval()

# # ── Tokenizer ───────────────────────────────────────────────────
# enc    = tiktoken.get_encoding("gpt2")
# EOS_ID = 50256

# prompt = "\n<human>\nWhat is the capital of France?\n<assistant>\n"
# ids    = enc.encode(prompt)
# x      = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

# print(f"Prompt: {prompt!r}")
# print("--- Generation ---")

# # ── Generation ──────────────────────────────────────────────────
# with torch.no_grad():
#     for _ in range(100):
#         logits, _ = model(x)
#         logits = logits[0, -1, :]                   # last position only

#         # temperature scale
#         logits = logits / 0.7

#         # top-k = 10 (not 50 — this model is small)
#         top_k = 10
#         v, _ = torch.topk(logits, top_k)
#         logits[logits < v[-1]] = -float("Inf")

#         probs   = torch.softmax(logits, dim=-1)
#         next_id = torch.multinomial(probs, num_samples=1)

#         x = torch.cat([x, next_id.view(1, 1)], dim=1)

#         if next_id.item() == EOS_ID:
#             print("\n[EOS]")
#             break
#         print(enc.decode([next_id.item()]), end="", flush=True)

# print("\n--- Generation End ---")

# with torch.no_grad():
#     for _ in range(60):
#         logits, _ = model(x)
#         next_id   = logits[0, -1, :].argmax()
#         x = torch.cat([x, next_id.view(1,1)], dim=1)
#         if next_id.item() == EOS_ID:
#             print("\n[EOS]")
#             break
#         print(enc.decode([next_id.item()]), end="", flush=True)