"""Generate a cited answer to a GST question using retrieved FAQ excerpts.

NOTE: llama-3.3-70b-versatile (used in the Apple 10-K project) was
deprecated by Groq (decommissioned Aug 16, 2026) and is now Enterprise-only.
This script uses openai/gpt-oss-120b instead, Groq's current recommended
general-purpose model. If the Apple 10-K project still references
llama-3.3-70b-versatile, it likely needs the same update.

Usage:
    python generate.py "What is Form GSTR-1?"
    python generate.py "How do I claim ITC?" --topic itc
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import retrieve as rt

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are a GST compliance assistant. Answer the user's question using ONLY "
    "the provided excerpts below — do not use outside knowledge. Every claim in "
    "your answer must be traceable to one of the excerpts, cited inline as "
    "(Source: <topic>/<section>). If the excerpts only partially answer the "
    "question, answer what they cover and explicitly say what's missing. If "
    "none of the excerpts are actually relevant, say you don't have a reliable "
    "source for this rather than guessing."
)


def build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[Excerpt {i}] Source: {r['topic']}/{r['section']}\n"
            f"Q: {r['question']}\nA: {r['answer']}"
        )
    return "\n\n".join(blocks)


def generate_answer(llm: ChatGroq, query: str, results: list[dict]) -> str:
    if not results:
        return (
            "I don't have a reliable source in the GST FAQ data for this question — "
            "you may want to check the official GST portal directly."
        )

    context = build_context(results)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {query}\n\nExcerpts:\n{context}"),
    ]
    response = llm.invoke(messages)
    return response.content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="The question to answer")
    parser.add_argument("--qdrant-url", default=rt.QDRANT_URL)
    parser.add_argument("--qdrant-api-key", default=rt.QDRANT_API_KEY)
    parser.add_argument("--top-k", type=int, default=rt.DEFAULT_TOP_K)
    parser.add_argument("--topic", default=None, help="Filter to one category, e.g. itc, returns, refund")
    parser.add_argument("--min-score", type=float, default=rt.MIN_SCORE)
    parser.add_argument("--model", default=GROQ_MODEL, help="Groq model ID to use for generation")
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. Export it first: export GROQ_API_KEY=your-key "
            "(same key/setup as the Apple 10-K project)."
        )

    device = rt.get_device()
    print(f"Loading {rt.MODEL_NAME} on device: {device} ...")
    embed_model = SentenceTransformer(rt.MODEL_NAME, device=device)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key)
    results = rt.search(
        client,
        embed_model,
        args.query,
        top_k=args.top_k,
        topic=args.topic,
        min_score=args.min_score,
    )

    llm = ChatGroq(model=args.model, temperature=0)
    answer = generate_answer(llm, args.query, results)

    print(f"\nQuestion: {args.query}\n")
    print("Answer:\n" + answer)

    if results:
        print("\nSources retrieved:")
        for r in results:
            print(f"  [{r['score']:.3f}] {r['topic']}/{r['section']} — {r['source_file']}")


if __name__ == "__main__":
    main()
