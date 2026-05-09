"""Integration tests for server-side embeddings + the unified /search endpoint.

Requires:
- A running Onecortex stack (gateway + auth + vector).
- The vector service started with ONECORTEX_VECTOR_EMBED_OPENAI_API_KEY set
  (the docker-compose.yml at the org root pulls it from $OPENAI_API_KEY).

These tests intentionally do NOT skip if the embedder isn't configured —
they fail loudly so misconfiguration surfaces immediately.
"""

import contextlib
import uuid

import pytest

from onecortex import (
    EmbedderSpec,
    GroupedQueryResult,
    TextRequiredError,
    ValuesAndTextConflictError,
)

# OpenAI text-embedding-3-small → 1536 dims.
OPENAI_MODEL = "text-embedding-3-small"
OPENAI_DIM = 1536


@pytest.fixture
def col_name():
    """Unique collection per test to keep them independent."""
    return f"sdk-embed-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def cleanup(oc_client, col_name):
    yield
    with contextlib.suppress(Exception):
        oc_client.vector.delete_collection(col_name)


@pytest.fixture
def embedder_collection(oc_client, col_name, cleanup):
    """A bm25-enabled collection bound to the OpenAI embedder."""
    oc_client.vector.create_collection(
        name=col_name,
        dimension=OPENAI_DIM,
        metric="cosine",
        bm25_enabled=True,
        embedder=EmbedderSpec(backend="openai", model=OPENAI_MODEL, input_type="document"),
    )
    return oc_client.vector.collection(col_name)


# ── Collection create with embedder ───────────────────────────────────────────


def test_create_collection_with_embedder_echoes_back(oc_client, col_name, cleanup):
    col = oc_client.vector.create_collection(
        name=col_name,
        dimension=OPENAI_DIM,
        embedder={"backend": "openai", "model": OPENAI_MODEL, "inputType": "document"},
    )
    assert col.embedder is not None
    assert col.embedder.backend == "openai"
    assert col.embedder.model == OPENAI_MODEL
    assert col.embedder.input_type == "document"

    # describe_collection should also surface the embedder.
    described = oc_client.vector.describe_collection(col_name)
    assert described.embedder is not None
    assert described.embedder.backend == "openai"


def test_create_collection_bm25_default_is_true(oc_client, col_name, cleanup):
    """Server defaults bm25Enabled to true on new collections (since v0.3)."""
    col = oc_client.vector.create_collection(name=col_name, dimension=OPENAI_DIM)
    assert col.bm25_enabled is True


def test_create_collection_bm25_explicit_false_preserved(oc_client, col_name, cleanup):
    col = oc_client.vector.create_collection(name=col_name, dimension=OPENAI_DIM, bm25_enabled=False)
    assert col.bm25_enabled is False


# ── Text-only upsert + query ──────────────────────────────────────────────────


def test_text_only_upsert_and_dense_query(embedder_collection):
    col = embedder_collection
    col.upsert(
        vectors=[
            {"id": "d1", "text": "the quick brown fox jumps over the lazy dog", "metadata": {"k": "fox"}},
            {"id": "d2", "text": "lorem ipsum dolor sit amet", "metadata": {"k": "lorem"}},
            {"id": "d3", "text": "machine learning models embed text into vector space", "metadata": {"k": "ml"}},
        ]
    )

    # Force dense path (hybrid=False) with a text query.
    result = col.query(text="a fox leaps over a sleepy dog", top_k=3, include_metadata=True)
    assert len(result.matches) >= 1
    assert result.matches[0].id == "d1"


def test_query_no_cache_does_not_error(embedder_collection):
    """noCache should bypass the LRU and still return results."""
    col = embedder_collection
    col.upsert(vectors=[{"id": "d1", "text": "hello world"}])
    r1 = col.query(text="hello", top_k=1, no_cache=True)
    r2 = col.query(text="hello", top_k=1, no_cache=True)
    assert r1.matches[0].id == "d1"
    assert r2.matches[0].id == "d1"


# ── /search endpoint ──────────────────────────────────────────────────────────


def test_search_text_dense_by_default(embedder_collection):
    col = embedder_collection
    col.upsert(
        vectors=[
            {"id": "a", "text": "apples are red", "metadata": {"category": "fruit"}},
            {"id": "b", "text": "bananas are yellow", "metadata": {"category": "fruit"}},
        ]
    )
    result = col.search(text="red apple", top_k=2, include_metadata=True)
    assert len(result.matches) >= 1
    assert result.matches[0].id == "a"


def test_search_hybrid_auto_detect(embedder_collection):
    """text + bm25Enabled collection ⇒ hybrid is auto-selected."""
    col = embedder_collection
    col.upsert(
        vectors=[
            {"id": "a", "text": "the quick brown fox", "metadata": {}},
            {"id": "b", "text": "machine learning embeddings", "metadata": {}},
            {"id": "c", "text": "a fox is a cunning animal", "metadata": {}},
        ]
    )
    # Auto-hybrid: no `hybrid` field, but text + bm25Enabled → hybrid path.
    result = col.search(text="fox", top_k=3, include_metadata=True)
    ids = {m.id for m in result.matches}
    assert "a" in ids or "c" in ids


def test_search_explicit_hybrid_object(embedder_collection):
    col = embedder_collection
    col.upsert(
        vectors=[
            {"id": "a", "text": "neural networks learn"},
            {"id": "b", "text": "fox jumps high"},
        ]
    )
    result = col.search(
        text="fox",
        top_k=2,
        hybrid={"alpha": 0.3, "bm25Weight": 0.7},
        include_metadata=True,
    )
    assert len(result.matches) >= 1


def test_search_with_dedup(embedder_collection):
    col = embedder_collection
    col.upsert(
        vectors=[
            {"id": "a1", "text": "apples are crisp", "metadata": {"docId": "doc-1"}},
            {"id": "a2", "text": "apples taste great", "metadata": {"docId": "doc-1"}},
            {"id": "b1", "text": "bananas are sweet", "metadata": {"docId": "doc-2"}},
        ]
    )
    # First-occurrence-wins dedup by docId — at most one result per docId.
    result = col.search(
        text="apple banana",
        top_k=10,
        dedup={"by": "docId"},
        include_metadata=True,
    )
    seen_doc_ids = [m.metadata.get("docId") for m in result.matches if m.metadata]
    assert len(seen_doc_ids) == len(set(seen_doc_ids))


def test_search_with_group_by_returns_grouped(embedder_collection):
    col = embedder_collection
    col.upsert(
        vectors=[
            {"id": "a1", "text": "apple pie", "metadata": {"category": "fruit"}},
            {"id": "a2", "text": "apple sauce", "metadata": {"category": "fruit"}},
            {"id": "b1", "text": "potato chips", "metadata": {"category": "snack"}},
            {"id": "b2", "text": "potato wedges", "metadata": {"category": "snack"}},
        ]
    )
    result = col.search(
        text="food",
        top_k=10,
        group_by={"field": "category", "limit": 5, "groupSize": 2},
    )
    assert isinstance(result, GroupedQueryResult)
    keys = {g.key for g in result.groups}
    assert keys.issubset({"fruit", "snack"})
    assert len(result.groups) >= 1


def test_search_explain_returns_plan_without_executing(embedder_collection):
    col = embedder_collection
    col.upsert(vectors=[{"id": "a", "text": "hello"}])
    result = col.search(text="hi", top_k=3, explain=True)
    assert isinstance(result, dict)
    assert "plan" in result
    plan = result["plan"]
    assert "source" in plan
    # Plan must NOT contain executed matches.
    assert "matches" not in plan


# ── New error codes ───────────────────────────────────────────────────────────


def test_values_and_text_conflict_on_embedder_collection(embedder_collection):
    col = embedder_collection
    with pytest.raises(ValuesAndTextConflictError) as exc_info:
        col.upsert(vectors=[{"id": "x", "values": [0.0] * OPENAI_DIM, "text": "hello"}])
    assert exc_info.value.code == "VALUES_AND_TEXT_CONFLICT"


def test_text_required_on_embedder_collection(embedder_collection):
    """Upsert with neither values nor text on an embedder-bound collection
    must surface as TEXT_REQUIRED."""
    col = embedder_collection
    with pytest.raises(TextRequiredError) as exc_info:
        col.upsert(vectors=[{"id": "x", "metadata": {"k": "v"}}])
    assert exc_info.value.code == "TEXT_REQUIRED"
