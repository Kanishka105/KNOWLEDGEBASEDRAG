
import pytest

from app.ai.rag.phase8_incremental_ingestion.deduplication import (
    DuplicateDetector,
    hash_bytes,
    hash_file,
    hash_text,
    normalize_text,
)


def test_hash_bytes_is_deterministic():
    assert hash_bytes(b"hello") == hash_bytes(b"hello")


def test_hash_file_matches_hash_bytes(tmp_path):
    file_path = tmp_path / "sample.pdf"
    content = b"sample PDF content"
    file_path.write_bytes(content)

    assert hash_file(file_path) == hash_bytes(content)


def test_normalized_text_ignores_case_and_whitespace():
    assert normalize_text(" Hello   RAG ") == "hello rag"
    assert hash_text("Hello   RAG") == hash_text(" hello rag ")


def test_duplicate_detector_tracks_unique_hashes():
    detector = DuplicateDetector()

    first = hash_text("Municipal awareness")
    duplicate = hash_text(" MUNICIPAL   AWARENESS ")
    second = hash_text("Water supply")

    assert detector.add(first) is True
    assert detector.add(duplicate) is False
    assert detector.add(second) is True
    assert detector.contains(first)
    assert len(detector) == 2


def test_check_and_add_reports_duplicate():
    detector = DuplicateDetector()
    content_hash = hash_text("RAG document")

    first_result = detector.check_and_add(content_hash)
    second_result = detector.check_and_add(content_hash)

    assert first_result["duplicate"] is False
    assert second_result["duplicate"] is True
    assert first_result["content_hash"] == second_result["content_hash"]
