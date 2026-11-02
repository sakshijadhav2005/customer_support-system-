"""FAQ retrieval for the support graph.

The base knowledge base is parsed from data/customer_support_knowledge_base.txt.
Chunks from files uploaded through the UI (see kb.py) are searched as well.
Search is hybrid: keyword matching and embedding search (Gemini + FAISS) are
merged with Reciprocal Rank Fusion. To change the retriever, replace
retrieve_entries() only; the graph and the retrieval node do not need to change.
"""
import hashlib
import json
import os
import re

import kb

KB_PATH = os.path.join(kb.DATA_DIR, kb.BASE_FILE_NAME)
INDEX_PATH = os.path.join(kb.DATA_DIR, "faiss.index")
META_PATH = os.path.join(kb.DATA_DIR, "faiss_meta.json")
TOP_K = 3
EMBEDDING_MODEL = "models/gemini-embedding-001"
MIN_SIMILARITY = 0.80  # cosine; below this an entry is treated as not relevant
RELATIVE_MARGIN = 0.04  # also drop entries this far below the best match
CANDIDATES = 10  # how many entries each method contributes to the fusion
RRF_K = 60  # Reciprocal Rank Fusion constant

STOP_WORDS = {
    "the", "is", "a", "an", "how", "do", "does", "did", "i", "my", "to", "of",
    "and", "or", "you", "your", "what", "can", "me", "it", "in", "on", "for",
    "are", "we", "with", "this", "that", "be", "have", "has", "if", "at", "by",
    "when", "where", "which", "who", "why", "there", "any", "about", "from",
    "as", "will", "would", "am", "was", "were", "so", "not",
}

_cache = {"mtime": None, "entries": []}
_index_cache = {"fingerprint": None, "index": None}
_embedder = None
last_method = ""  # "hybrid" or "keywords": which path served the latest query


def _words(text: str) -> set:
    """Lowercase, split into words, drop stop words, and fold simple plurals."""
    words = set()
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOP_WORDS:
            continue
        if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        words.add(word)
    return words


def parse_knowledge_base(path: str = KB_PATH) -> list:
    """Parse Q:/A: pairs into {question, answer, section} dicts.

    Capitalised "SECTION n: NAME" headings set the section for the entries
    below them. One question with its answer is one entry; entries are never split.
    """
    entries, section, current, field = [], "", None, None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("Q:"):
                current = {"question": line[2:].strip(), "answer": "", "section": section}
                entries.append(current)
                field = "question"
            elif line.startswith("A:") and current is not None:
                current["answer"] = line[2:].strip()
                field = "answer"
            elif not line:
                field = None
            elif re.fullmatch(r"(SECTION \d+:\s*)?[A-Z0-9 &/,'-]+", line) and field is None:
                section = line.split(":", 1)[1].strip() if ":" in line else line
                current = None
            elif current is not None and field:  # wrapped continuation line
                current[field] = f"{current[field]} {line}".strip()
    return entries


def load_entries() -> list:
    """Return the parsed base entries, re-parsing only if the file changed."""
    mtime = os.path.getmtime(KB_PATH) if os.path.exists(KB_PATH) else None
    if mtime != _cache["mtime"]:
        _cache["entries"] = parse_knowledge_base(KB_PATH) if mtime else []
        _cache["mtime"] = mtime
    return _cache["entries"]


def _format(entry: dict) -> str:
    return f"Q: {entry['question']}\nA: {entry['answer']}"


def _embedding_key() -> str:
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""


def _get_embedder():
    """Free Gemini embedding model (uses the same key as the chat model)."""
    global _embedder
    if _embedder is None:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _embedder = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL, google_api_key=_embedding_key()
        )
    return _embedder


def _all_entries() -> list:
    """Base entries plus uploaded chunks, as (text to embed, entry) pairs."""
    pairs = [(f"{e['section']}. Q: {e['question']} A: {e['answer']}", e) for e in load_entries()]
    pairs += [(f"{c['question']}. {c['answer']}", c) for c in kb.get_upload_entries()]
    return pairs


def _get_index(pairs: list):
    """Return the FAISS index for these entries, rebuilding only if they changed."""
    import faiss
    import numpy as np

    fingerprint = hashlib.sha256("\n".join(t for t, _ in pairs).encode("utf-8")).hexdigest()
    if _index_cache["fingerprint"] == fingerprint:
        return _index_cache["index"]

    index = None
    try:
        with open(META_PATH, encoding="utf-8") as f:
            if json.load(f).get("fingerprint") == fingerprint:
                index = faiss.read_index(INDEX_PATH)
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
        pass

    if index is None:
        texts = [t for t, _ in pairs]
        vectors = []
        for start in range(0, len(texts), 50):
            vectors += _get_embedder().embed_documents(texts[start:start + 50])
        matrix = np.array(vectors, dtype="float32")
        faiss.normalize_L2(matrix)
        index = faiss.IndexFlatIP(matrix.shape[1])  # inner product = cosine on unit vectors
        index.add(matrix)
        os.makedirs(kb.DATA_DIR, exist_ok=True)
        faiss.write_index(index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({"fingerprint": fingerprint, "count": len(texts)}, f)

    _index_cache.update(fingerprint=fingerprint, index=index)
    return index


def semantic_ranked(query: str, n: int = CANDIDATES) -> list:
    """Semantic search: entries above the similarity thresholds, best first."""
    import faiss
    import numpy as np

    if not query.strip():
        return []
    pairs = _all_entries()
    index = _get_index(pairs)
    vector = np.array([_get_embedder().embed_query(query)], dtype="float32")
    faiss.normalize_L2(vector)
    scores, ids = index.search(vector, min(n, len(pairs)))
    best = scores[0][0]
    return [
        pairs[i][1]
        for score, i in zip(scores[0], ids[0])
        if i >= 0 and score >= MIN_SIMILARITY and score >= best - RELATIVE_MARGIN
    ]


def keyword_ranked(query: str, n: int = CANDIDATES) -> list:
    """Keyword search: entries sharing the most words with the query, best first.

    Base entries are scored on section name + question; uploaded chunks are
    scored on their full text. Only entries tied for the highest score are
    kept, so single generic words do not pull in loosely related entries.
    """
    query_words = _words(query)
    scored = []
    for entry in load_entries():
        entry_words = _words(f"{entry['section']} {entry['question']}")
        scored.append((len(query_words & entry_words), entry))
    for chunk in kb.get_upload_entries():
        scored.append((len(query_words & _words(f"{chunk['question']} {chunk['answer']}")), chunk))
    scored = [pair for pair in scored if pair[0] > 0]
    if not scored:
        return []
    top = max(score for score, _ in scored)
    return [entry for score, entry in scored if score == top][:n]  # file order breaks ties


def keyword_retrieve(query: str, k: int = TOP_K) -> list:
    """Keyword-only retrieval, as formatted strings (used as the fallback)."""
    return [_format(e) for e in keyword_ranked(query)[:k]]


def hybrid_retrieve(query: str, k: int = TOP_K) -> list:
    """Hybrid search: fuse keyword and semantic rankings with Reciprocal Rank Fusion.

    Each entry scores sum(1 / (RRF_K + rank)) over the lists it appears in, so
    entries found by both methods rise to the top. An entry found by only one
    method can still be returned, which gives exact-term matches and
    paraphrase matches a chance.
    """
    fused, entries = {}, {}
    for ranking in (keyword_ranked(query), semantic_ranked(query)):
        for rank, entry in enumerate(ranking, start=1):
            key = entry["question"]
            entries[key] = entry
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
    best_first = sorted(fused, key=fused.get, reverse=True)
    return [_format(entries[key]) for key in best_first[:k]]


def retrieve_entries(query: str, k: int = TOP_K) -> list:
    """Return up to k entries as 'Q: ...\\nA: ...' strings, best first.

    Uses hybrid search (keywords + Gemini embeddings with FAISS). If no key is
    set or the embedding call fails (quota, network), falls back to keyword
    matching so the bot keeps working.
    """
    global last_method
    if _embedding_key():
        try:
            results = hybrid_retrieve(query, k)
            last_method = "hybrid"
            return results
        except Exception:
            pass
    last_method = "keywords"
    return keyword_retrieve(query, k)
