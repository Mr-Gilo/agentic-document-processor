"""
Document loading and text extraction.
Supports plain text and PDF input.
"""

import fitz  # PyMuPDF


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    text = ""
    for page_num, page in enumerate(doc):
        text += f"\n--- Page {page_num + 1} ---\n"
        text += page.get_text()
    doc.close()
    return text.strip()


def extract_text_from_string(text: str) -> str:
    """Clean and return plain text input."""
    return text.strip()