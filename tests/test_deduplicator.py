# data/tests/test_deduplicator.py

import pytest
from data.cleaning.deduplicator import MinHashDeduplicator


class TestMinHashDeduplicator:
    def test_first_doc_is_not_duplicate(self):
        """First document seen should never be a duplicate."""
        d = MinHashDeduplicator(threshold=0.8)
        assert d.is_duplicate("Hello world this is a test", "doc_0") is False

    def test_exact_duplicate_is_detected(self):
        """Identical text should be detected as duplicate."""
        d = MinHashDeduplicator(threshold=0.8)
        text = "The quick brown fox jumps over the lazy dog"
        d.is_duplicate(text, "doc_0")
        assert d.is_duplicate(text, "doc_1") is True

    def test_near_duplicate_is_detected(self):
        """Similar text with minor changes should be detected as duplicate with lenient threshold."""
        d = MinHashDeduplicator(threshold=0.5)
        text1 = "Climate change is one of the most pressing issues facing humanity today. " * 10
        text2 = "Climate change is one of the most critical issues facing humanity today. " * 10  # Minor change
        d.is_duplicate(text1, "doc_0")
        assert d.is_duplicate(text2, "doc_1") is True

    def test_different_docs_are_not_duplicates(self):
        """Completely different texts should not be duplicates."""
        d = MinHashDeduplicator(threshold=0.8)
        d.is_duplicate("Python is a programming language", "doc_0")
        d.is_duplicate("The weather is nice today", "doc_1")
        d.is_duplicate("Machine learning is fascinating", "doc_2")
        assert d.dedup_rate == 0.0

    def test_stats_tracks_counts(self):
        """Stats should correctly track seen and duplicate counts."""
        d = MinHashDeduplicator(threshold=0.8)
        text = "Sample text for testing"
        d.is_duplicate(text, "doc_0")
        d.is_duplicate(text, "doc_1")
        d.is_duplicate("Different text here", "doc_2")
        stats = d.stats()
        assert stats["total_seen"] == 3
        assert stats["duplicates_found"] == 1
        assert stats["unique_docs"] == 2
        assert stats["dedup_rate"] == round(1/3, 4)

    def test_dedup_rate_property(self):
        """dedup_rate should return correct ratio."""
        d = MinHashDeduplicator(threshold=0.8)
        text = "Duplicate text"
        d.is_duplicate(text, "doc_0")
        d.is_duplicate(text, "doc_1")
        d.is_duplicate(text, "doc_2")
        assert d.dedup_rate == 2/3

    def test_short_text_handling(self):
        """Should handle short documents."""
        d = MinHashDeduplicator(threshold=0.8)
        assert d.is_duplicate("Hi", "doc_0") is False
        assert d.is_duplicate("Hi", "doc_1") is True

    def test_threshold_affects_detection(self):
        """Lower threshold should detect more pairs as duplicates than higher threshold."""
        text1 = "The quick brown fox jumps over the lazy dog " * 20
        text2 = "The quick brown fox jumps over the lazy cat " * 20

        d_strict = MinHashDeduplicator(threshold=0.95)  # very strict
        d_strict.is_duplicate(text1, "doc_0")
        strict_result = d_strict.is_duplicate(text2, "doc_1")

        d_lenient = MinHashDeduplicator(threshold=0.5)  # very lenient
        d_lenient.is_duplicate(text1, "doc_0")
        lenient_result = d_lenient.is_duplicate(text2, "doc_1")

        # lenient threshold must catch at least as many as strict
        # (strict=False, lenient=True) OR both same — never strict catches more
        assert not (strict_result is True and lenient_result is False)

    def test_duplicate_doc_id_raises_error(self):
        """Calling is_duplicate with same doc_id twice should raise error."""
        d = MinHashDeduplicator(threshold=0.8)
        text = "Some unique text"
        d.is_duplicate(text, "doc_0")
        # MinHashLSH raises ValueError when inserting duplicate key
        with pytest.raises(ValueError):
            d.is_duplicate("Different text", "doc_0")

    def test_reset_clears_state(self):
        """Reset should clear the LSH index and reset all counters."""
        d = MinHashDeduplicator(threshold=0.8)
        text = "Some unique text"
        d.is_duplicate(text, "doc_0")
        d.is_duplicate(text, "doc_1")
        d.reset()
        assert d.stats()["total_seen"] == 0
        assert d.stats()["duplicates_found"] == 0
        assert d.stats()["unique_docs"] == 0
        assert d.stats()["dedup_rate"] == 0.0