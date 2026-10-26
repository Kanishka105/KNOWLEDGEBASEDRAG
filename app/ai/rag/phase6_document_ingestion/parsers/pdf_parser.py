from pathlib import Path
from pypdf import PdfReader


def extract_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF, preserving page numbers.

    Returns:
        [
            {"page_number": 1, "text": "..."},
            {"page_number": 2, "text": "..."}
        ]
    """
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported.")

    reader = PdfReader(str(path))

    if reader.is_encrypted:
        raise ValueError("Encrypted PDFs are not supported.")

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())

        if text:
            pages.append({
                "page_number": page_number,
                "text": text,
            })

    if not pages:
        raise ValueError(
            "No extractable text found. This may be a scanned PDF "
            "that needs OCR."
        )

    return pages
