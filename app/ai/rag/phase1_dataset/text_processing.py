
import re


def normalize_document_text(document: dict) -> str:
    """Combine a paper's title and body into searchable text."""
    title = document.get("title", "")
    body = document.get("text", document.get("abstract", ""))

    if isinstance(body, list):
        body = " ".join(str(sentence) for sentence in body)

    if not isinstance(body, str):
        body = ""

    if not isinstance(title, str):
        title = ""

    return f"{title} {body}".strip()


def tokenize(text: str) -> list[str]:
    """Lowercase text and extract alphanumeric word tokens."""
    return re.findall(r"\b\w+\b", text.lower())
