# model/loader.py

import torch
from model.gpt import GPT, GPTConfig

def load_model(ckpt_path: str, device: str = "cpu") -> tuple[GPT, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = GPT(GPTConfig(
        vocab_size=cfg.vocab_size,
        context_length=cfg.context_length,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        bias=True,
    ))
    state_dict = ckpt["model"]
    # Remove `_orig_mod.` prefix if it exists (happens if model was compiled)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, ckpt
