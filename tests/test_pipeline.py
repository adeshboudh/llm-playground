# tests/test_pipeline.py

import os
import yaml
import pytest
from pathlib import Path
from data.pipeline import run_pipeline


@pytest.fixture
def tmp_config(tmp_path):
    with open("data/configs/data_config.yaml") as f:
        cfg = yaml.safe_load(f)

    cfg["ingestion"]["num_samples"] = 20
    # as_posix() avoids Windows backslash escape corruption in YAML
    base = tmp_path.as_posix()
    cfg["artifacts"]["tokenizer_path"] = f"{base}/tokenizer"
    cfg["artifacts"]["shards_dir"] = f"{base}/shards"
    cfg["registry"]["push_shards"] = False

    config_path = f"{base}/test_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    return config_path, tmp_path


def test_pipeline_runs_end_to_end(tmp_config):
    config_path, tmp_path = tmp_config

    run_pipeline(config_path)

    # Tokenizer: save() writes a file, not a directory
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_json = tokenizer_dir / "tokenizer.json"
    assert tokenizer_dir.exists(), "Tokenizer directory was not created"
    assert tokenizer_json.exists(), "tokenizer.json not found"

    # Shards: at least one .bin file must exist with non-zero size
    shards_dir = tmp_path / "shards"
    assert shards_dir.exists(), "Shards directory was not created"
    shard_files = list(shards_dir.glob("*.bin"))
    assert len(shard_files) >= 1, f"Expected >=1 shard, got {len(shard_files)}"
    assert all(f.stat().st_size > 0 for f in shard_files), "Shard file is empty"


def test_pipeline_token_count_is_reasonable(tmp_config):
    """Replaces log-parsing test — verify output directly from shard files."""
    config_path, tmp_path = tmp_config

    run_pipeline(config_path)

    shards_dir = tmp_path / "shards"
    shard_files = list(shards_dir.glob("*.bin"))
    assert len(shard_files) >= 1

    # Each token is a uint16 (2 bytes) — check total tokens from file size
    import numpy as np
    total_tokens = sum(
        np.fromfile(str(f), dtype=np.uint16).shape[0] for f in shard_files
    )
    # 20 docs × ~800 tokens/doc avg → expect at least 5,000 tokens
    assert total_tokens >= 5_000, f"Too few tokens: {total_tokens} — pipeline may be broken"

