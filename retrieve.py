"""Query Qdrant for the most relevant GST FAQ Q&A pairs.

Usage:
    python retrieve.py "What is Form GSTR-1?"
    python retrieve.py "How do I claim ITC?" --topic itc --top-k 3
    python retrieve.py "some question" --include-review
"""

from __future__ import annotations

import argparse
import os

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# QDRANT_URL/QDRANT_API_KEY come from the environment (.env locally, HF Spaces
# secrets when deployed). Falls back to a local Docker instance with no key.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION_NAME = "gst_faq"
MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 5
# Below this cosine score, a match is too weak to trust as context for the
# LLM — better to tell the user "I don't have a good source for that" than
# let the model reach for a tangentially-related FAQ. 0.60 is a starting
# point based on the itc/returns spot-checks; tune it once you've tried more
# queries — too high and good-but-imperfect matches get dropped, too low and
# irrelevant ones leak through.
MIN_SCORE = 0.60

# bge-m3 needs no query-side instruction prefix (unlike bge-*-v1.5) — embed
# the raw question text, same as documents were embedded at ingest time.


def get_device() -> str:
    """Use CUDA (e.g. HF Spaces' ZeroGPU) or Apple's Metal backend when
    available, else CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_filter(topic: str | None, exclude_needs_review: bool) -> models.Filter | None:
    conditions = []
    if topic:
        conditions.append(models.FieldCondition(key="topic", match=models.MatchValue(value=topic)))
    if exclude_needs_review:
        conditions.append(models.FieldCondition(key="needs_review", match=models.MatchValue(value=False)))
    if not conditions:
        return None
    return models.Filter(must=conditions)


def search(
    client: QdrantClient,
    model: SentenceTransformer,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    topic: str | None = None,
    exclude_needs_review: bool = True,
    min_score: float = MIN_SCORE,
) -> list[dict]:
    vector = model.encode(query, normalize_embeddings=True).tolist()
    query_filter = build_filter(topic, exclude_needs_review)

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points

    # Drop weak matches rather than pass them to the LLM as if they were
    # good context — an irrelevant-but-included FAQ is worse than no match.
    results = [point for point in results if point.score >= min_score]

    return [
        {
            "score": point.score,
            "question": point.payload["question"],
            "answer": point.payload["answer"],
            "topic": point.payload["topic"],
            "section": point.payload["section"],
            "source_file": point.payload["source_file"],
            "needs_review": point.payload["needs_review"],
        }
        for point in results
    ]


def format_results(results: list[dict]) -> str:
    if not results:
        return "No matches cleared the similarity threshold — likely means this topic isn't well covered in the source FAQs."
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"{i}. [{r['score']:.3f}] ({r['topic']}/{r['section']}) {r['question']}\n"
            f"   {r['answer'][:200]}{'...' if len(r['answer']) > 200 else ''}\n"
            f"   source: {r['source_file']}"
        )
    return "\n\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The question to search for")
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--qdrant-api-key", default=QDRANT_API_KEY)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--topic", default=None, help="Filter to one category, e.g. itc, returns, refund")
    parser.add_argument("--min-score", type=float, default=MIN_SCORE, help="Drop matches below this cosine score")
    parser.add_argument(
        "--include-review",
        action="store_true",
        help="Include needs_review records in results (excluded by default).",
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Loading {MODEL_NAME} on device: {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    results = search(
        client,
        model,
        args.query,
        top_k=args.top_k,
        topic=args.topic,
        exclude_needs_review=not args.include_review,
        min_score=args.min_score,
    )

    print(f"\nTop {len(results)} matches for: {args.query!r} (min_score={args.min_score})\n")
    print(format_results(results))


if __name__ == "__main__":
    main()
