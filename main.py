"""
main.py — CLI entry point for the Hybrid RAG Pipeline.

Pipeline flow:
  User Query → retrieval/hybrid_search → llm/reranker → llm/gemini_client → Answer

Run:
  python main.py
"""

import sys
import os

# Windows: force UTF-8 output
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Add project root to path for module imports
ROOT = os.path.dirname(__file__)
sys.path.insert(0, ROOT)

from retrieval.hybrid_search import search
from llm.reranker import rerank, get_skip_stats
from llm.gemini_client import generate


def print_sep(char="=", width=60):
    print(char * width)


def run_pipeline(query: str):
    print_sep()
    print(f"❓ QUESTION: {query}")
    print_sep()

    # 1. Hybrid Retrieval
    print("\n🔍 [1/3] Hybrid search (Dense + BM25 + RRF)...")
    candidates = search(query)
    print(f"   → {len(candidates)} candidate chunks found.")

    if not candidates:
        print("[WARN] No results found. Run: python utils/helpers.py")
        return

    # 2. Reranking (with skip-rerank optimisation)
    print("\n🎯 [2/3] Reranking...")
    top_chunks = rerank(query, candidates)
    print(f"   → Top {len(top_chunks)} chunks selected.")

    # 3. Answer Generation
    print("\n✍️  [3/3] Generating answer (Gemini)...")
    result = generate(query, top_chunks)

    # Display results
    print_sep()
    print("📝 ANSWER:")
    print_sep("-")
    print(result["answer"])

    print_sep()
    print("📚 SOURCES:")
    print_sep("-")
    for source in result["sources"]:
        print(f"  [{source['number']}] {source['source']}")
        print(f"      {source['text_preview']}...")
        print()

    # Skip-rerank stats
    stats = get_skip_stats()
    print(f"⚡ Skip-rerank: {stats['skip_rate']}")
    print_sep()


def main():
    print_sep()
    print("   Hybrid RAG Pipeline — Q&A")
    print_sep()
    print("  Type 'q' or 'exit' to quit.\n")

    while True:
        try:
            query = input("Your question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            break

        if not query:
            continue
        if query.lower() in ("q", "exit", "quit"):
            print("Exiting...")
            break

        try:
            run_pipeline(query)
        except Exception as e:
            print(f"\n[ERROR] {e}")
            import traceback
            traceback.print_exc()

        print()


if __name__ == "__main__":
    main()
