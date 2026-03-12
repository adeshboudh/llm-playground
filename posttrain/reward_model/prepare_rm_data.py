from datasets import load_dataset
import numpy as np
import torch
import tiktoken
from torch.utils.data import Dataset
import glob, os

# Load preference dataset
ds = load_dataset("Anthropic/hh-rlhf")["train"].select(range(50000))

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]

def format_pair(example):
    chosen_text = example["chosen"]
    prompt_end = chosen_text.find("\n\nAssistant:")
    prompt = chosen_text[:prompt_end].strip()
    
    chosen   = example["chosen"].split("\n\nAssistant:")[1] + "\n<|endoftext|>"
    rejected = example["rejected"].split("\n\nAssistant:")[1] + "\n<|endoftext|>"
    
    # ✅ Add allowed_special to all three encode calls
    encoded_prompt   = enc.encode(prompt,   allowed_special={"<|endoftext|>"})
    encoded_chosen   = enc.encode(chosen,   allowed_special={"<|endoftext|>"})
    encoded_rejected = enc.encode(rejected, allowed_special={"<|endoftext|>"})
    
    input_ids = encoded_prompt + encoded_chosen + encoded_rejected
    
    labels = np.zeros(len(input_ids))
    chosen_start = len(encoded_prompt)
    labels[chosen_start:chosen_start + len(encoded_chosen)] = 1
    
    return {"input_ids": input_ids, "labels": labels}

# Tokenize
tokenized = ds.map(format_pair, remove_columns=ds.column_names)

# Save as fixed-length shards (pad to 1024)
SHARD_DIR = "/kaggle/working/rm_shards"
os.makedirs(SHARD_DIR, exist_ok=True)

def write_shards(split, data, shard_size=10_000_000):
    buf_ids, buf_labels = [], []
    shard_idx = 0
    
    def flush():
        nonlocal shard_idx
        path = os.path.join(SHARD_DIR, f"{split}_{shard_idx:05d}.npz")
        np.savez_compressed(path,
            ids    = np.array(buf_ids, dtype=np.int32),
            labels = np.array(buf_labels, dtype=np.int32),
        )
        print(f"  {split}_{shard_idx:05d}.npz  {len(buf_ids):,} tokens")
        shard_idx += 1
        return [], []
    
    for ex in data:
        ids, labels = ex["input_ids"], ex["labels"]
        buf_ids += ids
        buf_labels += list(labels)
        
        while len(buf_ids) >= shard_size:
            buf_ids, buf_labels = flush()
    
    if buf_ids:
        flush()

write_shards("rm_train", tokenized)
print("RM shards ready.")
