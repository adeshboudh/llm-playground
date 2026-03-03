# data/cleaning/filters.py

from abc import ABC, abstractmethod
from dataclasses import dataclass

# ── Base Filter Interface ──────────────────────────────────────

class BaseFilter(ABC):
    """All filters must implement this interface."""

    @abstractmethod
    def is_valid(self, text: str) -> bool:
        """Returns True if text passes the filter."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

# ── Concrete Filters ───────────────────────────────────────────

@dataclass
class LengthFilter(BaseFilter):
    """Filters documents based on character count."""

    min_length: int = 100
    max_length: int = 100_000

    @property
    def name(self): return "length_filter"

    def is_valid(self, text: str) -> bool:
        """Returns True if text length is within [min_length, max_length]."""
        return self.min_length <= len(text) <= self.max_length


@dataclass
class WordLengthFilter(BaseFilter):
    """Filters documents based on average word length."""

    min_avg: float = 3.0
    max_avg: float = 10.0

    @property
    def name(self): return "word_length_filter"

    def is_valid(self, text: str) -> bool:
        """Returns True if average word length is within [min_avg, max_avg]."""
        words = text.split()
        if not words:
            return False
        avg = sum(len(w) for w in words) / len(words)
        return self.min_avg <= avg <= self.max_avg


@dataclass
class SymbolRatioFilter(BaseFilter):
    """Filters documents based on symbol-to-word ratio."""

    max_ratio: float = 0.1   # symbols / words

    @property
    def name(self): return "symbol_ratio_filter"

    def is_valid(self, text: str) -> bool:
        """Returns True if symbol-to-word ratio does not exceed max_ratio."""
        words = text.split()
        if not words:
            return False
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        return (symbols / len(words)) <= self.max_ratio


@dataclass
class BulletLinesFilter(BaseFilter):
    """Filters documents with excessive bullet point lines."""

    max_ratio: float = 0.9

    @property
    def name(self): return "bullet_lines_filter"

    def is_valid(self, text: str) -> bool:
        """Returns True if bullet-line ratio does not exceed max_ratio."""
        lines = text.splitlines()
        if not lines:
            return False
        bullet_lines = sum(1 for l in lines if l.strip().startswith(("•", "-", "*", "·")))
        return (bullet_lines / len(lines)) <= self.max_ratio


@dataclass
class AlphanumericFilter(BaseFilter):
    """Filters documents based on alphanumeric character ratio."""

    min_ratio: float = 0.7   # alphanumeric chars / total chars

    @property
    def name(self): return "alphanum_filter"

    def is_valid(self, text: str) -> bool:
        """Returns True if alphanumeric ratio is at least min_ratio."""
        if not text:
            return False
        alphanum = sum(1 for c in text if c.isalnum())
        return (alphanum / len(text)) >= self.min_ratio


# ── Pipeline Orchestrator ──────────────────────────────────────

class FilterPipeline:
    """
    Runs a sequence of filters. Short-circuits on first failure.
    Tracks rejection stats per filter for observability.
    """

    def __init__(self, filters: list[BaseFilter]):
        self.filters = filters
        self.stats: dict[str, int] = {f.name: 0 for f in filters}
        self.total_seen = 0
        self.total_passed = 0

    def apply(self, text: str) -> bool:
        """
        Returns True if text passes ALL filters.
        Increments rejection counter for failing filter.
        """
        self.total_seen += 1
        for f in self.filters:
            if not f.is_valid(text):
                self.stats[f.name] += 1
                return False
        self.total_passed += 1
        return True

    def rejection_report(self) -> dict:
        return {
            "total_seen": self.total_seen,
            "total_passed": self.total_passed,
            "pass_rate": round(self.total_passed / max(self.total_seen, 1), 4),
            "rejections_by_filter": self.stats
        }