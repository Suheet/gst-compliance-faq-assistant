"""Tests for the GST Compliance FAQ Assistant pipeline.

These test the logic in retrieve.py, generate.py, and app.py — filtering,
score thresholds, prompt grounding, and the app's guard rails — without
needing a real Qdrant cluster, Groq API key, or downloaded embedding model.
External calls (Qdrant, the embedding model, the LLM) are mocked.

Run with:
    pytest test_app.py -v
"""

from unittest.mock import MagicMock

import generate as gen
import retrieve as rt
import app


# ---------------------------------------------------------------------------
# retrieve.py — filter construction
# ---------------------------------------------------------------------------

def test_build_filter_returns_none_with_no_conditions():
    assert rt.build_filter(topic=None, exclude_needs_review=False) is None


def test_build_filter_topic_only():
    filt = rt.build_filter(topic="itc", exclude_needs_review=False)
    assert filt is not None
    assert len(filt.must) == 1
    assert filt.must[0].key == "topic"


def test_build_filter_combines_topic_and_needs_review():
    filt = rt.build_filter(topic="itc", exclude_needs_review=True)
    assert len(filt.must) == 2
    keys = {condition.key for condition in filt.must}
    assert keys == {"topic", "needs_review"}


# ---------------------------------------------------------------------------
# retrieve.py — score threshold behavior
# ---------------------------------------------------------------------------

def _fake_point(score: float, **payload_overrides) -> MagicMock:
    payload = {
        "question": "q", "answer": "a", "topic": "returns",
        "section": "Form GSTR-1", "source_file": "f.pdf", "needs_review": False,
    }
    payload.update(payload_overrides)
    return MagicMock(score=score, payload=payload)


def _fake_client_and_model(points):
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=points)
    model = MagicMock()
    model.encode.return_value = MagicMock(tolist=lambda: [0.0] * 1024)
    return client, model


def test_search_drops_matches_below_min_score():
    client, model = _fake_client_and_model([_fake_point(0.75), _fake_point(0.40)])
    results = rt.search(client, model, "test query", min_score=0.6)
    assert len(results) == 1
    assert results[0]["score"] == 0.75


def test_search_returns_empty_when_everything_is_below_threshold():
    client, model = _fake_client_and_model([_fake_point(0.30)])
    results = rt.search(client, model, "test query", min_score=0.6)
    assert results == []


def test_search_keeps_matches_at_or_above_threshold():
    client, model = _fake_client_and_model([_fake_point(0.60)])
    results = rt.search(client, model, "test query", min_score=0.6)
    assert len(results) == 1


# ---------------------------------------------------------------------------
# generate.py — context building and grounding
# ---------------------------------------------------------------------------

def test_build_context_labels_each_excerpt_with_its_source():
    results = [
        {"score": 0.7, "topic": "returns", "section": "Form GSTR-1",
         "question": "What is Form GSTR-1?", "answer": "A statement of outward supplies.",
         "source_file": "f.pdf", "needs_review": False},
    ]
    context = gen.build_context(results)
    assert "[Excerpt 1]" in context
    assert "Source: returns/Form GSTR-1" in context
    assert "Q: What is Form GSTR-1?" in context
    assert "A: A statement of outward supplies." in context


def test_generate_answer_never_calls_llm_when_no_results():
    fake_llm = MagicMock()
    answer = gen.generate_answer(fake_llm, "some question", [])
    assert not fake_llm.invoke.called
    assert "don't have a reliable source" in answer.lower()


def test_generate_answer_grounds_the_prompt_in_retrieved_excerpts():
    results = [
        {"score": 0.7, "topic": "returns", "section": "Form GSTR-1",
         "question": "What is GSTR-1?", "answer": "A statement of outward supplies.",
         "source_file": "f.pdf", "needs_review": False},
    ]
    fake_llm = MagicMock()
    fake_llm.invoke.return_value = MagicMock(content="Form GSTR-1 is ... (Source: returns/Form GSTR-1)")

    answer = gen.generate_answer(fake_llm, "What is GSTR-1?", results)

    assert fake_llm.invoke.called
    system_msg, human_msg = fake_llm.invoke.call_args[0][0]
    assert "ONLY" in system_msg.content  # system prompt forbids outside knowledge
    assert "returns/Form GSTR-1" in human_msg.content  # excerpt source made it into the prompt
    assert answer == "Form GSTR-1 is ... (Source: returns/Form GSTR-1)"


# ---------------------------------------------------------------------------
# app.py — guard rails and response formatting
# ---------------------------------------------------------------------------

def test_respond_blocks_when_groq_key_missing(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("QDRANT_URL", "https://fake.qdrant.io")
    result = app.respond("What is GSTR-1?", [], "All", 5, 0.6)
    assert "GROQ_API_KEY" in result


def test_respond_blocks_when_qdrant_url_missing(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    result = app.respond("What is GSTR-1?", [], "All", 5, 0.6)
    assert "QDRANT_URL" in result


def test_respond_appends_sources_when_results_found(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("QDRANT_URL", "https://fake.qdrant.io")
    monkeypatch.setattr(app, "get_resources", lambda: ("embedder", "client", "llm"))
    monkeypatch.setattr(rt, "search", lambda *a, **k: [
        {"score": 0.8, "topic": "returns", "section": "Form GSTR-1", "question": "q",
         "answer": "a", "source_file": "/x/GSTR 1.pdf", "needs_review": False}
    ])
    monkeypatch.setattr(gen, "generate_answer", lambda *a, **k: "Form GSTR-1 is a statement of outward supplies.")

    result = app.respond("What is GSTR-1?", [], "All", 5, 0.6)

    assert "Sources:" in result
    assert "returns/Form GSTR-1" in result
    assert "GSTR 1.pdf" in result


def test_respond_omits_sources_section_when_nothing_retrieved(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake-key")
    monkeypatch.setenv("QDRANT_URL", "https://fake.qdrant.io")
    monkeypatch.setattr(app, "get_resources", lambda: ("embedder", "client", "llm"))
    monkeypatch.setattr(rt, "search", lambda *a, **k: [])
    monkeypatch.setattr(gen, "generate_answer", lambda *a, **k: "I don't have a reliable source for this.")

    result = app.respond("unrelated question", [], "All", 5, 0.6)

    assert "Sources:" not in result
