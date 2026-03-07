"""
HellaSwag evaluation for our custom GPT model.
Downloads hellaswag_val.jsonl on first use → model/eval/hellaswag/ (gitignored).

Eval method (completion style):
  For each (ctx + ending_i), compute cross-entropy loss over ending tokens only.
  acc      → argmin(sum_loss)   raw loss
  acc_norm → argmin(avg_loss)   length-normalized loss  ← primary metric
"""

import os
import json
import requests
from tqdm import tqdm
import torch
import torch.nn.functional as F

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

_URLS = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val":   "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test":  "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}


def _download(split: str) -> None:
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    dest = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if os.path.exists(dest):
        return
    url = _URLS[split]
    print(f"Downloading {url} ...")
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(total=total, unit="iB", unit_scale=True) as bar:
        for chunk in resp.iter_content(1024):
            f.write(chunk)
            bar.update(len(chunk))


def iterate_examples(split: str = "val"):
    _download(split)
    path = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)


def render_example(example: dict, tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    """
    Tokenize one HellaSwag example.

    Returns:
        tokens : (4, max_len) LongTensor  — ctx + each of 4 endings
        mask   : (4, max_len) LongTensor  — 1 only over ending tokens
        label  : int                      — correct ending index (0-3)
    """
    ctx_tokens = tokenizer.encode(example["ctx"])
    label = example["label"]

    tok_rows, mask_rows = [], []
    for end in example["endings"]:
        end_tokens = tokenizer.encode(" " + end)  # space prefix for clean subword boundary
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))

    max_len = max(len(r) for r in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask   = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)]  = torch.tensor(mask_row)

    return tokens, mask, label


@torch.no_grad()
def evaluate_hellaswag(
    model,
    tokenizer,
    device: str,
    split: str = "val",
    max_examples: int | None = None,
) -> dict:
    """
    Evaluate model on HellaSwag.

    Args:
        model        : GPT instance — forward must return (logits, loss)
        tokenizer    : BPETokenizerTrainer — must expose .encode(str) -> list[int]
        device       : 'cuda' | 'cpu' | 'mps'
        split        : 'val' (default) | 'train'
        max_examples : cap for quick mid-training checks (e.g. 200); None = full eval

    Returns:
        {"acc": float, "acc_norm": float, "num_total": int}
    """
    model.eval()
    num_correct = num_correct_norm = num_total = 0

    for example in iterate_examples(split):
        if max_examples is not None and num_total >= max_examples:
            break

        tokens, mask, label = render_example(example, tokenizer)
        tokens = tokens.to(device)
        mask   = mask.to(device)

        logits, _ = model(tokens)              # (4, max_len, vocab_size)

        shift_logits = logits[:, :-1, :].contiguous()   # (4, max_len-1, V)
        shift_tokens = tokens[:, 1:].contiguous()        # (4, max_len-1)
        shift_mask   = mask[:, 1:].contiguous()          # (4, max_len-1)

        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_tokens = shift_tokens.view(-1)
        losses = F.cross_entropy(flat_logits, flat_tokens, reduction="none")
        losses = losses.view(4, -1)                      # (4, max_len-1)

        masked_losses = losses * shift_mask
        sum_loss = masked_losses.sum(dim=1)              # (4,)
        avg_loss = sum_loss / shift_mask.sum(dim=1)      # (4,) normalized by ending length

        pred      = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        num_total        += 1
        num_correct      += int(pred == label)
        num_correct_norm += int(pred_norm == label)

    return {
        "acc":       num_correct / num_total,
        "acc_norm":  num_correct_norm / num_total,
        "num_total": num_total,
    }
