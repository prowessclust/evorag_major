"""Print retrieval quality diagnostics for an EvoRAG query."""

import argparse

from config import RETRIEVAL_TOP_K
from retriever import retrieval_diagnostics


def main() -> None:
    """Run retrieval diagnostics from the command line."""
    parser = argparse.ArgumentParser(description="Inspect EvoRAG retrieval diversity.")
    parser.add_argument("query", help="Question or search query to diagnose.")
    parser.add_argument("--top-k", type=int, default=RETRIEVAL_TOP_K)
    args = parser.parse_args()

    diagnostics = retrieval_diagnostics(args.query, top_k=args.top_k)

    print("\n" + "=" * 60)
    print("  EvoRAG Retrieval Diagnostics")
    print("=" * 60)
    print(f"Query                    : {diagnostics['query']}")
    print(f"Requested top_k          : {diagnostics['requested_top_k']}")
    print(f"Candidate chunks checked : {diagnostics['candidate_count']}")
    print(f"Returned chunks          : {diagnostics['returned_count']}")
    print(f"Unique documents         : {diagnostics['unique_document_count']}")
    print(f"Unique sources           : {diagnostics['unique_source_count']}")
    print(f"Duplicate candidate ratio: {diagnostics['duplicate_candidate_ratio']}")
    print(f"Average retrieval score  : {diagnostics['average_retrieval_score']}")

    print("\nTop retrieved results:")
    for item in diagnostics["results"]:
        print(
            f"  [{item['rank']}] score={item['score']} | "
            f"{item['source']} | {item['title'][:70]}"
        )
        print(f"      {item['url']}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
