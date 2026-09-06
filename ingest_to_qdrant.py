"""Embed qdrant_documents.jsonl with BAAI/bge-m3 and upsert into Qdrant.

Usage:
    python ingest_to_qdrant.py
    python ingest_to_qdrant.py --input data/qdrant_documents.jsonl --recreate
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

load_dotenv()

# Let unsupported ops silently fall back to CPU instead of crashing the run,
# since MPS operator coverage still has gaps compared to CUDA.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "qdrant_documents.jsonl"

# QDRANT_URL/QDRANT_API_KEY come from the environment (.env locally, HF Spaces
# secrets when deployed). Falls back to a local Docker instance with no key,
# so nothing changes for local-only runs.
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")  # None is fine for local/no-auth
COLLECTION_NAME = "gst_faq"
MODEL_NAME = "BAAI/bge-m3"
VECTOR_SIZE = 1024
BATCH_SIZE = 32  # smaller batches — uploading to cloud over the network is slower/less reliable than local disk writes

# Unlike bge-*-v1.5, bge-m3 does not need a query-side instruction prefix —
# BAAI's own docs say it "no longer requires adding instructions to the
# queries." Embed queries and documents identically at retrieval time.


def get_device() -> str:
    """Use the M-series GPU via Metal (MPS) when available, else CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_documents(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def ensure_collection(client: QdrantClient, name: str, vector_size: int, recreate: bool) -> None:
    exists = client.collection_exists(collection_name=name)
    if exists and recreate:
        client.delete_collection(collection_name=name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        # Qdrant Cloud requires an explicit payload index before a field can
        # be used in a filter (local/self-hosted Qdrant allows unindexed
        # filtering, just slower on large collections). retrieve.py filters
        # on both of these, so both need an index.
        client.create_payload_index(
            collection_name=name, field_name="topic", field_schema=models.PayloadSchemaType.KEYWORD
        )
        client.create_payload_index(
            collection_name=name, field_name="needs_review", field_schema=models.PayloadSchemaType.BOOL
        )


def batched(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def upsert_documents(client: QdrantClient, model: SentenceTransformer, records: list[dict]) -> int:
    total = 0
    for batch in batched(records, BATCH_SIZE):
        texts = [record["document"] for record in batch]
        # normalize_embeddings=True pairs with Distance.COSINE above.
        vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        points = [
            models.PointStruct(
                id=record["id"],
                vector=vector.tolist(),
                payload=record["payload"],
            )
            for record, vector in zip(batch, vectors)
        ]

        # Cloud upserts occasionally hit transient timeouts; retry a couple
        # times with backoff before giving up on this batch.
        for attempt in range(3):
            try:
                client.upsert(collection_name=COLLECTION_NAME, points=points)
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"  batch failed ({exc.__class__.__name__}), retrying in {wait}s...")
                time.sleep(wait)

        total += len(points)
        print(f"  upserted {total}/{len(records)}")
    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--qdrant-url", default=QDRANT_URL)
    parser.add_argument("--qdrant-api-key", default=QDRANT_API_KEY)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before ingesting (simplest way to handle edited records).",
    )
    args = parser.parse_args()

    records = load_documents(args.input.resolve())
    if not records:
        print(f"No documents found in {args.input}")
        return

    device = get_device()
    print(f"Loading {MODEL_NAME} on device: {device} ...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    client = QdrantClient(url=args.qdrant_url, api_key=args.qdrant_api_key, timeout=60)
    ensure_collection(client, args.collection, VECTOR_SIZE, args.recreate)

    print(f"Embedding and upserting {len(records)} documents into '{args.collection}' ...")
    upsert_documents(client, model, records)

    count = client.count(collection_name=args.collection, exact=True).count
    print(f"Done. Collection '{args.collection}' now has {count} points.")


if __name__ == "__main__":
    main()
