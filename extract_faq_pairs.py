"""Extract FAQ question/answer pairs from the GST PDF collection.

Usage:
    python extract_faq_pairs.py
    python extract_faq_pairs.py --input data/returns --output data/qa_pairs

The extractor recognises both PDF formats currently in this project:
  * current GST portal exports (large section headings, bold numbered questions)
  * legacy GSTN documents (decimal question numbers such as 2.1)

Records marked ``needs_review`` should be checked before they are embedded.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pdfplumber


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DEFAULT_INPUT / "qa_pairs"

PORTAL_QUESTION = re.compile(r"^(\d+)\.\s+(.*)")
LEGACY_QUESTION = re.compile(r"^(\d+(?:\.\d+)+)\s+(.*)")
WHITESPACE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE.sub(" ", text).strip()


def append(record: dict, field: str, text: str) -> None:
    record[field] = normalize(f"{record[field]} {text}")


def page_lines(page):
    """Return text lines with their character formatting metadata."""
    return page.dedupe_chars(tolerance=1).extract_text_lines(
        x_tolerance=2, y_tolerance=3, return_chars=True
    ) or []


def is_portal_faq(pdf) -> bool:
    """The current portal export identifies the FAQ family in its first pages."""
    for page in pdf.pages[:2]:
        for line in page_lines(page):
            if "FAQs >" in line["text"]:
                return True
    return False


def is_question_style(chars: list[dict]) -> bool:
    if not chars:
        return False
    sizes = [char["size"] for char in chars]
    fonts = [char["fontname"] for char in chars]
    return min(sizes) >= 13 and all("Arial-Bold" in font for font in fonts)


def extract_portal(pdf, source: Path) -> list[dict]:
    pairs, current = [], None
    section = source.stem

    for page_no, page in enumerate(pdf.pages, start=1):
        for line in page_lines(page):
            text = normalize(line["text"])
            chars = [char for char in line["chars"] if not char["text"].isspace()]
            if not text:
                continue
            if text.startswith(source.stem + " ") or text.startswith("https://tutorial.gst.gov.in/"):
                continue

            sizes = [char["size"] for char in chars]
            # Browser-export title/date headers and page footers are 7pt, while
            # FAQ content begins at 10.4pt. Removing them also preserves a
            # question that continues on the next page.
            if max(sizes, default=0) <= 8:
                continue
            if max(sizes, default=0) >= 18 and "FAQs >" in text:
                section = text.replace("FAQs >", "").strip()
                continue

            match = PORTAL_QUESTION.match(text)
            if match and is_question_style(chars):
                if current:
                    pairs.append(current)
                current = {
                    "id": f"{source.stem}::{match.group(1)}",
                    "source_file": str(source),
                    "section": section,
                    "question_number": match.group(1),
                    "question": match.group(2),
                    "answer": "",
                    "page_start": page_no,
                    "page_end": page_no,
                }
            elif is_question_style(chars) and current and not current["answer"]:
                append(current, "question", text)
                current["page_end"] = page_no
            elif current:
                append(current, "answer", text)
                current["page_end"] = page_no

    if current:
        pairs.append(current)
    return pairs


def extract_legacy(pdf, source: Path) -> list[dict]:
    """Extract decimal-numbered questions in the older GSTN FAQ documents."""
    pairs, current = [], None
    section = source.stem
    content_started = False

    for page_no, page in enumerate(pdf.pages, start=1):
        for line in page_lines(page):
            text = normalize(line["text"])
            # Each legacy PDF begins with a multi-page table of contents. Its
            # entries resemble questions, so start only at the real introduction.
            if text == "1. Introduction":
                content_started = True
                continue
            if not content_started:
                continue
            if not text or re.search(r"\.{3,}", text):  # table-of-contents leader dots
                continue

            match = LEGACY_QUESTION.match(text)
            # Legacy documents have no reliable distinct question font. Requiring a
            # question mark avoids version-history/table-of-contents numeric lines.
            if match and "?" in text:
                if current:
                    pairs.append(current)
                number, question = match.groups()
                current = {
                    "id": f"{source.stem}::{number}",
                    "source_file": str(source),
                    "section": section,
                    "question_number": number,
                    "question": question,
                    "answer": "",
                    "page_start": page_no,
                    "page_end": page_no,
                }
            elif current:
                append(current, "answer", text)
                current["page_end"] = page_no

    if current:
        pairs.append(current)
    return pairs


def add_review_flags(pairs: list[dict]) -> None:
    for pair in pairs:
        pair["needs_review"] = (
            len(pair["answer"]) < 30
            or "?" not in pair["question"]
            or not pair["answer"]
        )


def output_path(pdf_path: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / pdf_path.relative_to(input_root).with_suffix(".json")


def process_pdf(pdf_path: Path, input_root: Path, output_root: Path) -> tuple[Path, int, int] | None:
    with pdfplumber.open(pdf_path) as pdf:
        portal = is_portal_faq(pdf)
        legacy = "faq" in pdf_path.name.lower()
        if not portal and not legacy:
            return None
        pairs = extract_portal(pdf, pdf_path) if portal else extract_legacy(pdf, pdf_path)

    add_review_flags(pairs)
    destination = output_path(pdf_path, input_root, output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination, len(pairs), sum(pair["needs_review"] for pair in pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    input_root = args.input.resolve()
    output_root = args.output.resolve()
    for pdf_path in sorted(input_root.rglob("*.pdf")):
        result = process_pdf(pdf_path, input_root, output_root)
        if result:
            destination, count, review_count = result
            print(f"{pdf_path.name}: {count} pairs ({review_count} need review) -> {destination}")


if __name__ == "__main__":
    main()
