"""Extracts raw text from a CV PDF so it can be fed to the LLM."""

import pdfplumber


def extract_cv_text(pdf_path: str) -> str:
    """Return the full plain-text content of a CV PDF.

    Raises FileNotFoundError / pdfplumber errors upstream if the file is
    missing or unreadable — caller should handle that.
    """
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                chunks.append(text)

    full_text = "\n".join(chunks).strip()

    if not full_text:
        raise ValueError(
            f"Could not extract any text from '{pdf_path}'. "
            "It may be a scanned/image-only PDF — try exporting the CV "
            "as a text-based PDF instead."
        )

    return full_text
