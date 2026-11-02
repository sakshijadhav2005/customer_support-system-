"""Chunks from documents uploaded through the UI.

Uploaded documents are indexed in data/knowledge.json. The retriever calls
get_upload_entries() on every query, so changes are visible immediately.
The base knowledge base file is handled by retriever.py, not here.
"""
import json
import os

BASE_FILE_NAME = "customer_support_knowledge_base.txt"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INDEX_PATH = os.path.join(DATA_DIR, "knowledge.json")


def _load() -> list:
    try:
        with open(INDEX_PATH, encoding="utf-8") as f:
            return json.load(f).get("documents", [])
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(documents: list) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump({"documents": documents}, f, indent=2)


def list_documents() -> list:
    return [{k: v for k, v in d.items() if k != "chunks"} | {"chunk_count": len(d["chunks"])}
            for d in _load()]


def add_document(name: str, file_type: str, size: int, added: str, chunks: list) -> None:
    documents = [d for d in _load() if d["name"] != name]
    documents.append(
        {"name": name, "type": file_type, "size": size, "added": added, "chunks": chunks}
    )
    _save(documents)


def remove_document(name: str) -> None:
    _save([d for d in _load() if d["name"] != name])
    path = os.path.join(DATA_DIR, name)
    if os.path.isfile(path):
        os.remove(path)


def get_upload_entries() -> list:
    """Chunks of uploaded documents, as entry dicts."""
    entries = []
    for doc in _load():
        for i, chunk in enumerate(doc["chunks"], 1):
            entries.append(
                {
                    "question": f"{doc['name']} (section {i})",
                    "answer": chunk,
                    "tags": [],
                    "source": doc["name"],
                }
            )
    return entries
