import os, glob
import numpy as np
import tiktoken
from datasets import load_dataset

SHARD_SIZE = 10_000_000
OUT_DIR    = "/kaggle/working/ultrachat_shards"
CACHE_DIR  = "/kaggle/working/data/ultrachat"
os.makedirs(OUT_DIR, exist_ok=True)

enc       = tiktoken.get_encoding("gpt2")
EOS_ID    = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

# These are the ONLY two headers — match exactly in training AND inference
HUMAN_HDR = enc.encode("\n<|user|>\n")       # compact, unambiguous
ASST_HDR  = enc.encode("\n<|assistant|>\n")  # compact, unambiguous

def apply_chat_template(messages):
    ids, labels = [], []
    for msg in messages:
        if msg["role"] == "user":
            tokens  = HUMAN_HDR + enc.encode(msg["content"])
            ids    += tokens
            labels += [-100] * len(tokens)           # mask entire user turn
        elif msg["role"] == "assistant":
            body   = enc.encode(msg["content"])
            tokens = ASST_HDR + body + [EOS_ID]
            ids    += tokens
            labels += [-100] * len(ASST_HDR)         # mask header
            labels += body + [EOS_ID]                # train on body + EOS only
    return ids, labels

def write_shards(split_name, hf_split):
    print(f"\nProcessing {split_name} ({len(hf_split)} rows)...")
    buf_ids, buf_labels = [], []
    shard_idx, total   = 0, 0

    def flush(b_ids, b_labels, idx):
        path = os.path.join(OUT_DIR, f"{split_name}_{idx:05d}.npz")
        np.savez_compressed(
            path,
            ids    = np.array(b_ids,    dtype=np.int32),
            labels = np.array(b_labels, dtype=np.int32),
        )
        print(f"  shard {idx:05d} → {len(b_ids):,} tokens → {path}")
        return idx + 1

    for row in hf_split:
        i, l = apply_chat_template(row["messages"])
        buf_ids += i; buf_labels += l; total += len(i)
        while len(buf_ids) >= SHARD_SIZE:
            shard_idx = flush(buf_ids[:SHARD_SIZE], buf_labels[:SHARD_SIZE], shard_idx)
            buf_ids   = buf_ids[SHARD_SIZE:]
            buf_labels= buf_labels[SHARD_SIZE:]

    if buf_ids:
        shard_idx = flush(buf_ids, buf_labels, shard_idx)
    print(f"  {split_name}: {total:,} tokens, {shard_idx} shards")

train_raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", cache_dir=CACHE_DIR)
val_raw   = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft",  cache_dir=CACHE_DIR)
write_shards("train", train_raw)
write_shards("val",   val_raw)
print("\nDone.")
