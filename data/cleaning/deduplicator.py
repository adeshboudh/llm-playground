# data/cleaning/deduplicator.py

from datasketch import MinHash, MinHashLSH
import re


class MinHashDeduplicator:
    """
    Near-duplicate detection using MinHash + LSH.
    Operates in streaming fashion — no need to hold all docs in RAM.

    Jaccard similarity threshold: docs above this are considered duplicates.
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.85):
        """
        Args:
            num_perm: Number of permutations for MinHash (higher = more accurate, slower)
            threshold: Jaccard similarity threshold for duplicate detection
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._seen_count = 0
        self._dup_count = 0

    def _shingle(self, text: str, k: int = 5) -> set[str]:
        """Generates k-character shingles from text."""
        text = re.sub(r'\s+', ' ', text.lower())
        return {text[i:i+k] for i in range(len(text) - k + 1)}

    def _make_minhash(self, text: str) -> MinHash:
        """Creates a MinHash signature for the given text."""
        m = MinHash(num_perm=self.num_perm)
        for shingle in self._shingle(text):
            m.update(shingle.encode('utf-8'))
        return m

    def is_duplicate(self, text: str, doc_id: str) -> bool:
        """
        Returns True if text is a near-duplicate of a previously seen doc.
        If not duplicate, registers this doc in the LSH index.

        Args:
            text: Document text
            doc_id: Unique identifier (e.g., index or URL hash)

        Returns:
            bool: True if duplicate, False if unique (and registered)
        """
        self._seen_count += 1
        mh = self._make_minhash(text)
        results = self.lsh.query(mh)
        if results:
            self._dup_count += 1
            return True
        self.lsh.insert(doc_id, mh)
        return False

    @property
    def dedup_rate(self) -> float:
        """Returns the ratio of documents detected as duplicates."""
        return self._dup_count / max(self._seen_count, 1)

    def stats(self) -> dict:
        """Returns deduplication statistics."""
        return {
            "total_seen": self._seen_count,
            "duplicates_found": self._dup_count,
            "unique_docs": self._seen_count - self._dup_count,
            "dedup_rate": round(self.dedup_rate, 4)
        }

    def reset(self) -> None:
        """Clears the LSH index and resets all counters."""
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        self._seen_count = 0
        self._dup_count = 0