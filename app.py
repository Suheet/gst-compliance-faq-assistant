"""Gradio chat UI for the GST Compliance FAQ Assistant.

Run locally:
    python app.py

This is also the entrypoint for the Hugging Face Space deployment (Gradio SDK).
"""

import os

import gradio as gr
import spaces
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

import generate as gen
import retrieve as rt

load_dotenv()

TOPICS = ["All", "itc", "payments", "refund", "registration", "returns"]

# Loaded lazily on first question, then cached for the life of the process —
# same intent as Streamlit's @st.cache_resource, just done by hand since
# Gradio has no built-in equivalent.
_embed_model = None
_qdrant_client = None
_llm = None


def get_resources():
    global _embed_model, _qdrant_client, _llm
    if _embed_model is None:
        device = rt.get_device()
        _embed_model = SentenceTransformer(rt.MODEL_NAME, device=device)
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(url=rt.QDRANT_URL, api_key=rt.QDRANT_API_KEY, timeout=60)
    if _llm is None:
        _llm = ChatGroq(model=gen.GROQ_MODEL, temperature=0)
    return _embed_model, _qdrant_client, _llm


@spaces.GPU
def respond(message: str, history, topic_choice: str, top_k: float, min_score: float) -> str:
    if not os.environ.get("GROQ_API_KEY"):
        return "⚠️ GROQ_API_KEY is not set. Add it to this Space's secrets and restart."
    if not os.environ.get("QDRANT_URL"):
        return "⚠️ QDRANT_URL is not set. Add it to this Space's secrets and restart."

    embed_model, client, llm = get_resources()
    topic = None if topic_choice == "All" else topic_choice
    results = rt.search(client, embed_model, message, top_k=int(top_k), topic=topic, min_score=min_score)
    answer = gen.generate_answer(llm, message, results)

    if results:
        sources = "\n\n**Sources:**\n" + "\n".join(
            f"- [{r['score']:.3f}] {r['topic']}/{r['section']} — `{r['source_file'].split('/')[-1]}`"
            for r in results
        )
        answer += sources
    return answer


demo = gr.ChatInterface(
    fn=respond,
    title="🧾 GST Compliance FAQ Assistant",
    description=(
        "Answers are generated only from official GST FAQ documents (returns, ITC, "
        "refunds, registration, payments) — every claim is traceable to a cited source, "
        "and the assistant says so explicitly when it doesn't have a reliable one."
    ),
    additional_inputs=[
        gr.Dropdown(choices=TOPICS, value="All", label="Category"),
        gr.Slider(minimum=1, maximum=10, value=rt.DEFAULT_TOP_K, step=1, label="Sources to consider"),
        gr.Slider(minimum=0.0, maximum=1.0, value=rt.MIN_SCORE, step=0.01, label="Minimum similarity score"),
    ],
    examples=[
        ["What is Form GSTR-1?", "All", 5, 0.6],
        ["How do I claim input tax credit?", "itc", 5, 0.6],
        ["Can I get a refund if I paid excess tax?", "refund", 5, 0.6],
    ],
)

if __name__ == "__main__":
    demo.launch()
