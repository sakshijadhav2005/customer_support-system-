"""Upload pipeline: validate, save, extract text, chunk, add to the knowledge base."""
import io
import os
import re
from datetime import datetime

import kb

ALLOWED_TYPES = {".pdf", ".txt", ".docx"}
MAX_BYTES = 5 * 1024 * 1024
CHUNK_WORDS = 250
OVERLAP_WORDS = 50


class UploadError(Exception):
    """Carries a message that is safe to show to the user."""


def extract_text(ext: str, data: bytes) -> str:
    if ext == ".txt":
        return data.decode("utf-8", errors="replace")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    from docx import Document

    return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)


def chunk_text(text: str, size: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS) -> list:
    words = text.split()
    chunks, step = [], size - overlap
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return chunks


def process_upload(filename: str, data: bytes) -> int:
    """Returns the number of chunks added, or raises UploadError."""
    name = re.sub(r"[^\w.\- ]", "_", os.path.basename(filename))
    ext = os.path.splitext(name)[1].lower()
    if name == kb.BASE_FILE_NAME:
        raise UploadError("This file name is reserved for the base knowledge base. Please rename the file")
    if ext not in ALLOWED_TYPES:
        raise UploadError("This file type is not supported. Please upload PDF, TXT or DOCX")
    if len(data) > MAX_BYTES:
        raise UploadError("This file is larger than the allowed limit. Please upload a smaller file")

    try:
        text = extract_text(ext, data)
    except Exception:
        raise UploadError("The file could not be processed. Please try another file")
    chunks = chunk_text(text)
    if not text.strip():
        raise UploadError("No readable text was found. A scanned PDF may need OCR")

    try:
        os.makedirs(kb.DATA_DIR, exist_ok=True)
        with open(os.path.join(kb.DATA_DIR, name), "wb") as f:
            f.write(data)
        kb.add_document(name, ext.lstrip(".").upper(), len(data),
                        datetime.now().strftime("%Y-%m-%d %H:%M"), chunks)
    except Exception:
        raise UploadError("The file could not be processed. Please try another file")
    return len(chunks)
