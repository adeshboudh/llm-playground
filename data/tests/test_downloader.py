# tests/test_downloader.py
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.ingestion.downloader import DatasetDownloader, DownloaderConfig


def make_config(**overrides):
    base = dict(
        source="HuggingFaceFW/fineweb-edu",
        subset="sample-10BT",
        num_samples=3,
        text_column="text",
        streaming=True
    )
    base.update(overrides)
    return DownloaderConfig(**base)


def test_validate_config_rejects_zero_samples():
    with pytest.raises(ValueError, match="num_samples"):
        DatasetDownloader(make_config(num_samples=0))


def test_validate_config_rejects_empty_text_column():
    with pytest.raises(ValueError, match="text_column"):
        DatasetDownloader(make_config(text_column=""))


def test_iterate_before_load_raises():
    d = DatasetDownloader(make_config())
    with pytest.raises(RuntimeError, match="load()"):
        list(d.iterate())


def test_iterate_yields_correct_count():
    d = DatasetDownloader(make_config(num_samples=3)).load()
    results = list(d.iterate())
    assert len(results) == 3


def test_iterate_yields_non_empty_strings():
    d = DatasetDownloader(make_config(num_samples=3)).load()
    for text in d.iterate():
        assert isinstance(text, str)
        assert len(text) > 0


def test_repr_contains_source():
    d = DatasetDownloader(make_config())
    assert "fineweb-edu" in repr(d)
    assert "3" in repr(d)


def test_len_matches_config():
    d = DatasetDownloader(make_config(num_samples=3))
    assert len(d) == 3
