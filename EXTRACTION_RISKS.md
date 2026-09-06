## Known risks & limitations — `q_a_test.py`

This extractor relies on visual/structural heuristics from the PDF rather than a real document schema, so it's tied to how this specific GST tutorial site exports PDFs. Documenting these so future re-runs (new PDF versions, other FAQ pages) get sanity-checked instead of trusted blindly.

### 1. Question detection depends on font styling
A line is only treated as a new question if it's bold (`Arial-Bold`) and ≥13pt, matched against `^\d+\.\s+`. If the source PDF is regenerated with different fonts/sizes (e.g., a browser update changes the export), questions will silently fall through and get merged into the previous answer instead of raising an error.

### 2. Section headers require an exact pattern
New sections are only detected when a line is ≥18pt **and** contains the literal string `"FAQs >"`. Any section heading that doesn't match this (different phrasing, smaller size) won't update `section`, and its questions will be silently attributed to the previous section.

### 3. Sub-topic detection is inferred, not read from the document
Many real sub-topics in the PDF (e.g., "My Master Facility", "Editing E-invoice Details") have no detectable heading at all — they're plain bold text below the font-size threshold. The script infers a new sub-topic only when question numbering resets or goes non-increasing (`qnum <= last_question_number`). This works for this PDF but is a heuristic:
- A genuinely out-of-order or mis-numbered question (typo in the source, e.g., skipping from 3 to 3 again) would trigger a false sub-topic split.
- A sub-topic that happens to continue numbering upward from the previous one (no reset) would be silently merged into the prior sub-topic.

### 4. Header/footer stripping is hard-coded to this domain
Lines are dropped only if they start with `"GSTR 1 "` or `"https://tutorial.gst.gov.in/"`. Pulling a different FAQ page (different form, different site) will need these literals updated or the repeating header/footer will leak into question/answer text.

### 5. `needs_review` flag is a coarse heuristic
Currently flags: answer under 30 characters, missing `?` in the question, or empty answer. This catches obviously broken records but won't catch content that's wrong-but-well-formed — e.g., an answer accidentally merged from two different questions, or a question that got truncated mid-sentence but still happens to end in `?`.

### 6. No automated uniqueness check on output
IDs are unique by construction given the current heuristics, but there's no assertion that fails loudly if two records ever produce the same ID (e.g., from an edge case in #3). Right now a collision would silently overwrite the JSON entry rather than error out.

### 7. Hard-coded absolute file paths
`INPUT_PDF` / `OUTPUT_JSON` are local machine paths (`/Users/suheetsonawane/...`), so the script isn't portable/runnable elsewhere without editing them manually.

### 8. Not tested against tables, images, or multi-column layouts
This PDF is plain single-column text. If a future source FAQ page includes tables, images, or two-column layouts, `extract_text_lines` won't reconstruct reading order correctly and answers could come out jumbled or incomplete.

---
**Suggested mitigation (not yet implemented):** add an assertion after extraction that raises if any `id` in `pairs` is duplicated, so a heuristic failure surfaces immediately instead of producing silently-wrong data.
