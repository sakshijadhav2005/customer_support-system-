"""Checks for retriever.py. Run: python test_retriever.py"""
import re
import sys

from dotenv import load_dotenv

load_dotenv()  # so the Gemini key is available for embedding retrieval

import retriever  # noqa: E402


def show(query: str, results: list) -> None:
    print(f"\nQuery: {query!r} -> {len(results)} result(s)")
    for r in results:
        print("  " + r.replace("\n", "\n  ")[:200])


def main() -> int:
    failures = []

    def check(name: str, ok: bool) -> None:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            failures.append(name)

    entries = retriever.parse_knowledge_base()
    with open(retriever.KB_PATH, encoding="utf-8") as f:
        q_lines = sum(1 for line in f if line.startswith("Q:"))
    check("file loads and has entries", len(entries) > 0)
    check(f"parsed entries ({len(entries)}) match Q lines ({q_lines})", len(entries) == q_lines)
    check("every entry has an answer and a section",
          all(e["answer"] and e["section"] for e in entries))

    query = "What is your return policy?"
    results = retriever.retrieve_entries(query)
    show(query, results)
    check("return policy entry retrieved", any(re.search(r"return policy", r, re.I) for r in results[:1]))

    query = "Do you have physical stores?"
    results = retriever.retrieve_entries(query)
    show(query, results)
    check("physical store entry retrieved first",
          bool(results) and results[0].startswith("Q: Do you have a physical store?"))

    query = "What is the stock price of your company?"
    results = retriever.retrieve_entries(query)
    show(query, results)
    check("unrelated query returns empty list", results == [])

    check("at most 3 results", len(retriever.retrieve_entries("shipping order return account")) <= 3)
    check("empty query returns empty list", retriever.retrieve_entries("") == [])

    print(f"\nRetrieval method used for the last query: {retriever.last_method}")
    for query in ["can I get my money back", "my parcel is late"]:
        show(query, retriever.retrieve_entries(query))

    print("\nAll checks passed." if not failures else f"\n{len(failures)} check(s) failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
