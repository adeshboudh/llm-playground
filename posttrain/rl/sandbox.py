# posttrain/rl/sandbox.py
import torch
from posttrain.sft.infer_sft import load_sft_model, generate
from posttrain.reward_model.infer_rm import load_rm, score

def run_sandbox(prompt: str, num_samples: int = 4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sft_model, sft_tok, _  = load_sft_model(device=device)
    rm_model,  rm_tok,  _  = load_rm(device=device)

    # Generate N responses from SFT model
    responses = [
        generate(prompt, sft_model, sft_tok, device, max_new_tokens=80)
        for _ in range(num_samples)
    ]

    # Score each with RM
    scored = [
        (score(prompt, r, rm_model, rm_tok, device), r)
        for r in responses
    ]
    ranked = sorted(scored, key=lambda x: x[0], reverse=True)

    print(f"\nPrompt: {prompt}\n{'='*60}")
    for i, (s, r) in enumerate(ranked):
        print(f"\n[Rank {i+1} | score={s:+.4f}]\n{r}")

    print(f"\n{'='*60}")
    print(f"Best:  {ranked[0][1][:80]}...")
    print(f"Worst: {ranked[-1][1][:80]}...")
    return ranked

if __name__ == "__main__":
    run_sandbox("What is the capital of France?")
    run_sandbox("Explain what machine learning is.")
    run_sandbox("Write a Python function to reverse a string.")
