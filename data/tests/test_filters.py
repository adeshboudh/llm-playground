# tests/test_filters.py
import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from data.cleaning.filters import (
    BaseFilter,
    LengthFilter,
    WordLengthFilter,
    SymbolRatioFilter,
    BulletLinesFilter,
    AlphanumericFilter,
    FilterPipeline,
)


# ── LengthFilter Tests ────────────────────────────────────────

class TestLengthFilter:
    def test_rejects_text_below_min_length(self):
        f = LengthFilter(min_length=100, max_length=1000)
        assert f.is_valid("short text") is False

    def test_rejects_text_above_max_length(self):
        f = LengthFilter(min_length=10, max_length=100)
        assert f.is_valid("a" * 200) is False

    def test_passes_text_within_bounds(self):
        f = LengthFilter(min_length=10, max_length=100)
        assert f.is_valid("a" * 50) is True

    def test_passes_text_at_min_boundary(self):
        f = LengthFilter(min_length=10, max_length=100)
        assert f.is_valid("a" * 10) is True

    def test_passes_text_at_max_boundary(self):
        f = LengthFilter(min_length=10, max_length=100)
        assert f.is_valid("a" * 100) is True

    def test_name_property(self):
        f = LengthFilter()
        assert f.name == "length_filter"


# ── WordLengthFilter Tests ────────────────────────────────────

class TestWordLengthFilter:
    def test_rejects_low_avg_word_length(self):
        f = WordLengthFilter(min_avg=3.0, max_avg=10.0)
        # "a a a" -> avg word length = 1
        assert f.is_valid("a a a a a") is False

    def test_rejects_high_avg_word_length(self):
        f = WordLengthFilter(min_avg=3.0, max_avg=10.0)
        # Very long words
        assert f.is_valid("antidisestablishmentarianism supercalifragilisticexpialidocious") is False

    def test_passes_normal_avg_word_length(self):
        f = WordLengthFilter(min_avg=3.0, max_avg=10.0)
        # Normal English text
        assert f.is_valid("The quick brown fox jumps over the lazy dog") is True

    def test_rejects_empty_text(self):
        f = WordLengthFilter()
        assert f.is_valid("") is False

    def test_rejects_whitespace_only(self):
        f = WordLengthFilter()
        assert f.is_valid("   ") is False

    def test_name_property(self):
        f = WordLengthFilter()
        assert f.name == "word_length_filter"


# ── SymbolRatioFilter Tests ───────────────────────────────────

class TestSymbolRatioFilter:
    def test_rejects_high_symbol_ratio(self):
        f = SymbolRatioFilter(max_ratio=0.1)
        # Many symbols relative to words
        assert f.is_valid("hello !@#$% ^&*() world") is False

    def test_passes_low_symbol_ratio(self):
        f = SymbolRatioFilter(max_ratio=0.1)
        # Normal text with few symbols
        assert f.is_valid("The quick brown fox jumps over the lazy dog") is True

    def test_rejects_empty_text(self):
        f = SymbolRatioFilter()
        assert f.is_valid("") is False

    def test_rejects_whitespace_only(self):
        f = SymbolRatioFilter()
        assert f.is_valid("   ") is False

    def test_name_property(self):
        f = SymbolRatioFilter()
        assert f.name == "symbol_ratio_filter"


# ── BulletLinesFilter Tests ───────────────────────────────────

class TestBulletLinesFilter:
    def test_rejects_excessive_bullet_lines(self):
        f = BulletLinesFilter(max_ratio=0.5)
        # More than 50% bullet lines
        text = "- item 1\n- item 2\n- item 3\nnormal line"
        assert f.is_valid(text) is False

    def test_passes_normal_text(self):
        f = BulletLinesFilter(max_ratio=0.9)
        text = "This is a normal paragraph.\nAnother line here.\nAnd one more."
        assert f.is_valid(text) is True

    def test_passes_acceptable_bullet_ratio(self):
        f = BulletLinesFilter(max_ratio=0.5)
        # Only 25% bullet lines
        text = "Line one\nLine two\nLine three\n- bullet item"
        assert f.is_valid(text) is True

    def test_rejects_empty_text(self):
        f = BulletLinesFilter()
        assert f.is_valid("") is False

    def test_detects_various_bullet_styles(self):
        f = BulletLinesFilter(max_ratio=0.3)
        text = "• bullet 1\n* bullet 2\n- bullet 3\nnormal line"
        assert f.is_valid(text) is False

    def test_name_property(self):
        f = BulletLinesFilter()
        assert f.name == "bullet_lines_filter"


# ── AlphanumericFilter Tests ──────────────────────────────────

class TestAlphanumericFilter:
    def test_rejects_low_alphanumeric_ratio(self):
        f = AlphanumericFilter(min_ratio=0.7)
        # Lots of symbols, low alphanumeric
        assert f.is_valid("!@#$%^&*()_+{}[]|\\:;\"'<>,.?/") is False

    def test_passes_normal_text(self):
        f = AlphanumericFilter(min_ratio=0.7)
        text = "The quick brown fox jumps over the lazy dog"
        assert f.is_valid(text) is True

    def test_passes_at_boundary(self):
        f = AlphanumericFilter(min_ratio=0.5)
        # 66.7% alphanumeric
        assert f.is_valid("abc123!!!") is True

    def test_rejects_empty_text(self):
        f = AlphanumericFilter()
        assert f.is_valid("") is False

    def test_name_property(self):
        f = AlphanumericFilter()
        assert f.name == "alphanum_filter"


# ── FilterPipeline Tests ───────────────────────────────────────

class TestFilterPipeline:
    def test_passes_when_all_filters_pass(self):
        pipeline = FilterPipeline([
            LengthFilter(min_length=5, max_length=100),
            WordLengthFilter(min_avg=2.0, max_avg=10.0),
        ])
        assert pipeline.apply("The quick brown fox") is True

    def test_fails_when_one_filter_fails(self):
        pipeline = FilterPipeline([
            LengthFilter(min_length=100, max_length=1000),  # Will fail
            WordLengthFilter(min_avg=2.0, max_avg=10.0),
        ])
        assert pipeline.apply("Short text") is False

    def test_short_circuits_on_first_failure(self):
        pipeline = FilterPipeline([
            LengthFilter(min_length=100, max_length=1000),
            WordLengthFilter(min_avg=2.0, max_avg=10.0),
        ])
        pipeline.apply("short")  # Fails on length
        # Only length_filter should have a rejection
        assert pipeline.stats["length_filter"] == 1
        assert pipeline.stats["word_length_filter"] == 0

    def test_tracks_rejection_stats(self):
        pipeline = FilterPipeline([LengthFilter(min_length=100)])
        pipeline.apply("short")  # Rejected
        pipeline.apply("a" * 150)  # Passed
        pipeline.apply("tiny")  # Rejected
        report = pipeline.rejection_report()
        assert report["total_seen"] == 3
        assert report["total_passed"] == 1
        assert report["pass_rate"] == round(1/3, 4)  # 0.3333

    def test_rejection_report_shows_filter_name(self):
        pipeline = FilterPipeline([LengthFilter(min_length=100)])
        pipeline.apply("short")
        report = pipeline.rejection_report()
        assert "length_filter" in report["rejections_by_filter"]
        assert report["rejections_by_filter"]["length_filter"] == 1

    def test_handles_empty_filter_list(self):
        pipeline = FilterPipeline([])
        assert pipeline.apply("any text") is True
        assert pipeline.total_passed == 1