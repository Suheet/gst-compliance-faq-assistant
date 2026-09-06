"""Validate, normalize, and combine extracted FAQ pairs for Qdrant ingestion.

This script does not call Qdrant or create embeddings. It produces JSONL records
with a stable UUID, embedding-ready ``document`` text, and a filterable
``payload``. Your ingestion script can embed ``document`` and upload ``id``,
the resulting vector, and ``payload`` to Qdrant.

Usage:
    python prepare_for_qdrant.py
    python prepare_for_qdrant.py --include-review-records
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import uuid
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "qa_pairs"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "qdrant_documents.jsonl"
DEFAULT_REJECTED = PROJECT_ROOT / "data" / "qdrant_rejected.jsonl"
NAMESPACE = uuid.UUID("36f915f1-45ca-4f13-8e24-a44748b9107e")
WHITESPACE = re.compile(r"\s+")
REQUIRED_FIELDS = ("id", "question", "answer", "source_file", "page_start", "page_end")


def normalize(value: object) -> str:
    return WHITESPACE.sub(" ", str(value or "")).strip()


def topic_from_path(json_file: Path, input_root: Path) -> str:
    """Use the first folder beneath qa_pairs as the high-level GST topic."""
    relative = json_file.relative_to(input_root)
    return relative.parts[0] if len(relative.parts) > 1 else "general"


def stable_point_id(record_id: str) -> str:
    """Same source record always maps to the same Qdrant-compatible UUID."""
    return str(uuid.uuid5(NAMESPACE, record_id))


def build_document(question: str, answer: str) -> str:
    return f"Question: {question}\nAnswer: {answer}"


def validate(raw: dict) -> list[str]:
    errors = [field for field in REQUIRED_FIELDS if not normalize(raw.get(field))]
    if not isinstance(raw.get("page_start"), int) or not isinstance(raw.get("page_end"), int):
        errors.append("page numbers must be integers")
    elif raw["page_start"] > raw["page_end"]:
        errors.append("page_start must not exceed page_end")
    return errors


def prepare_record(raw: dict, json_file: Path, input_root: Path) -> dict:
    question = normalize(raw["question"])
    answer = normalize(raw["answer"])
    source_path = Path(raw["source_file"])
    source_name = source_path.name
    source_category = topic_from_path(json_file, input_root)
    record_id = normalize(raw["id"])
    checksum = hashlib.sha256(build_document(question, answer).encode("utf-8")).hexdigest()
    # Question numbers restart in some PDF sections, so the extractor's record_id
    # is useful metadata but is not globally unique. Include the JSON-relative
    # path and content checksum to make the Qdrant point ID stable and unique.
    point_key = f"{json_file.relative_to(input_root)}::{record_id}::{checksum}"

    return {
        "id": stable_point_id(point_key),
        "document": build_document(question, answer),
        "payload": {
            "record_id": record_id,
            "question": question,
            "answer": answer,
            "topic": source_category,
            "section": normalize(raw.get("section")) or source_name,
            "question_number": normalize(raw.get("question_number")),
            "source_file": str(source_path),
            "document_name": source_name,
            "page_start": raw["page_start"],
            "page_end": raw["page_end"],
            "needs_review": bool(raw.get("needs_review", False)),
            "content_sha256": checksum,
        },
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rejected-output", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument(
        "--include-review-records",
        action="store_true",
        help="Include records previously marked needs_review (excluded by default).",
    )
    args = parser.parse_args()

    input_root = args.input.resolve()
    prepared, rejected = [], []
    counts: Counter[str] = Counter()

    for json_file in sorted(input_root.rglob("*.json")):
        raw_records = json.loads(json_file.read_text(encoding="utf-8"))
        if not isinstance(raw_records, list):
            rejected.append({"input_file": str(json_file), "errors": ["expected a JSON array"]})
            continue

        for index, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                rejected.append({"input_file": str(json_file), "index": index, "errors": ["expected an object"]})
                continue
            errors = validate(raw)
            if errors:
                rejected.append({"input_file": str(json_file), "index": index, "record": raw, "errors": errors})
                continue
            if raw.get("needs_review") and not args.include_review_records:
                rejected.append({
                    "input_file": str(json_file),
                    "index": index,
                    "record": raw,
                    "errors": ["needs_review is true; rerun with --include-review-records to include"],
                })
                continue

            prepared.append(prepare_record(raw, json_file, input_root))
            counts[topic_from_path(json_file, input_root)] += 1

    write_jsonl(args.output.resolve(), prepared)
    write_jsonl(args.rejected_output.resolve(), rejected)
    print(f"Prepared {len(prepared)} Qdrant documents: {args.output.resolve()}")
    print(f"Excluded/rejected {len(rejected)} records: {args.rejected_output.resolve()}")
    print("By topic: " + ", ".join(f"{topic}={count}" for topic, count in sorted(counts.items())))


if __name__ == "__main__":
    main()
