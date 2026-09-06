import json
import re
import pdfplumber

INPUT_PDF = "/Users/suheetsonawane/Desktop/India Vapsi/Github/Projects/GST Compliance RAG Assistant/data/returns/GSTR 1.pdf"
OUTPUT_JSON = "/Users/suheetsonawane/Desktop/India Vapsi/Github/Projects/GST Compliance RAG Assistant/data/returns/gstr1_qa_pairs.json"

question_start = re.compile(r"^(\d+)\.\s+(.*)")
pairs = []
current = None
section = "Form GSTR-1"

def normalize(text):
    return re.sub(r"\s+", " ", text).strip()

def append(field, text):
    current[field] = normalize(f"{current[field]} {text}")

with pdfplumber.open(INPUT_PDF) as pdf:
    for page_no, page in enumerate(pdf.pages, start=1):
        lines = (
            page
            .dedupe_chars(tolerance=1)
            .extract_text_lines(
                x_tolerance=2,
                y_tolerance=3,
                return_chars=True,
            )
        )

        for line in lines:
            text = normalize(line["text"])
            chars = [c for c in line["chars"] if not c["text"].isspace()]

            if not text:
                continue

            # Remove repeated browser-export header/footer.
            if (
                text.startswith("GSTR 1 ")
                or text.startswith("https://tutorial.gst.gov.in/")
            ):
                continue

            sizes = [char["size"] for char in chars]
            fonts = [char["fontname"] for char in chars]

            is_question_style = (
                bool(chars)
                and min(sizes) >= 13
                and all("Arial-Bold" in font for font in fonts)
            )

            # Optional: maintain a section label when FAQ section headings appear.
            if max(sizes, default=0) >= 18 and "FAQs >" in text:
                section = text.replace("FAQs >", "").strip()
                continue

            match = question_start.match(text)

            # A bold 14pt numbered line begins a new Q&A pair.
            if match and is_question_style:
                if current:
                    pairs.append(current)

                current = {
                    "id": f"{section}::{match.group(1)}",
                    "section": section,
                    "question_number": int(match.group(1)),
                    "question": match.group(2),
                    "answer": "",
                    "page_start": page_no,
                    "page_end": page_no,
                }

            # A bold 14pt unnumbered line directly after a question continues it.
            elif is_question_style and current and not current["answer"]:
                append("question", text)
                current["page_end"] = page_no

            # Normal 10.4pt text belongs to the current answer.
            elif current:
                append("answer", text)
                current["page_end"] = page_no

if current:
    pairs.append(current)

# Flag records for manual review, rather than silently accepting poor chunks.
for pair in pairs:
    pair["needs_review"] = (
        len(pair["answer"]) < 30
        or "?" not in pair["question"]
        or not pair["answer"]
    )

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(pairs, f, ensure_ascii=False, indent=2)

print(f"Extracted {len(pairs)} Q&A pairs")