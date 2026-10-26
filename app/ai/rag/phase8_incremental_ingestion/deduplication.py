
"""Content hashing and duplicate detection for Phase 8 ingestion."""

import hashlib
import re
from pathlib import Path
from typing import Any


def hash_bytes(content: bytes) -> str:
    """Return the SHA-256 hash of raw file bytes."""
    return hashlib.sha256(content).hexdigest()


def hash_file(file_path: str | Path) -> str:
    """Hash a file without loading the entire file into memory."""
    digest = hashlib.sha256()

    with Path(file_path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def normalize_text(text: str) -> str:
    """Normalize whitespace for comparison of extracted document text."""
    text = text.casefold()
    return re.sub(r"\s+", " ", text).strip()


def hash_text(text: str) -> str:
    """Hash normalized extracted text."""
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class DuplicateDetector:
    """Track content hashes and identify previously seen documents."""

    def __init__(self, existing_hashes: set[str] | None = None):
        self._hashes = set(existing_hashes or set())

    def contains(self, content_hash: str) -> bool:
        """Check whether a content hash is already registered."""
        return content_hash in self._hashes

    def add(self, content_hash: str) -> bool:
        """Register a hash; return False if it was already present."""
        if content_hash in self._hashes:
            return False

        self._hashes.add(content_hash)
        return True

    def check_and_add(self, content_hash: str) -> dict[str, Any]:
        """Register new content and report whether it is a duplicate."""
        added = self.add(content_hash)

        return {
            "duplicate": not added,
            "content_hash": content_hash,
        }

    def __len__(self) -> int:
        return len(self._hashes)
