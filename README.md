---
title: GST Compliance FAQ Assistant
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
short_description: Source-cited RAG assistant for Indian GST compliance FAQs
---

# GST Compliance FAQ Assistant

A source-anchored RAG assistant that answers questions about Indian GST (Goods
and Services Tax) compliance — returns, ITC, refunds, registration, and
payments — grounded strictly in official GST FAQ documents. Every answer cites
the specific FAQ section it came from, and the assistant explicitly says when
it doesn't have a reliable source rather than guessing.

**[Live demo](https://huggingface.co/spaces/Suheet/gst-compliance-faq-assistant)**

## How it works

1. **Extraction**: FAQ PDFs from the official GST tutorial site are parsed
   with `pdfplumber`, using font size/style heuristics to detect question
   boundaries, section headers, and unheaded sub-topics.
2. **Embedding**: Rather than splitting documents into arbitrary fixed-size
   chunks (the common RAG default), each extracted Q&A pair is embedded as a
   single, semantically complete unit with `BAAI/bge-m3`. This avoids the
   classic fixed-size-chunking failure mode where a question and its answer
   get split across chunk boundaries, or an answer gets truncated mid-thought.
   Each point is stored in Qdrant with category, section, and page-range
   metadata for filtering.
3. **Retrieval**: A user's question is embedded the same way and matched
   against the collection, with a minimum similarity threshold — weak matches
   are dropped before ever reaching the LLM.
4. **Generation**: Retrieved excerpts are passed to an LLM (Groq's
   `openai/gpt-oss-120b`) with a system prompt that requires every claim to be
   traceable to a cited excerpt, and forbids filling gaps from general
   knowledge.

## Stack

- **Vector DB**: Qdrant Cloud (free tier)
- **Embeddings**: BAAI/bge-m3 (sentence-transformers)
- **LLM**: Groq (openai/gpt-oss-120b) via LangChain
- **UI**: Gradio
- **Extraction**: pdfplumber

## Running locally

```bash
pip install -r requirements.txt
```

Create a `.env` file with:
```
GROQ_API_KEY=your-groq-key
QDRANT_URL=your-qdrant-cluster-url
QDRANT_API_KEY=your-qdrant-api-key
```

Then:
```bash
python app.py
```
