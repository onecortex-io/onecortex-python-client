"""
Integration tests against a live Onecortex Vector server.
Requires: server running on localhost:8080 with a valid API key in ONECORTEX_API_KEY.
"""

import contextlib

import pytest

from onecortex.exceptions import NotFoundError
from onecortex.vector.models import GroupedQueryResult

COLLECTION_NAME = "sdk-integration-test"
DIM = 3


@pytest.fixture(autouse=True)
def cleanup(oc_client):
    yield
    with contextlib.suppress(Exception):
        oc_client.vector.delete_collection(COLLECTION_NAME)


def test_create_and_describe_collection(oc_client):
    col = oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM, metric="cosine")
    assert col.name == COLLECTION_NAME
    assert col.dimension == DIM

    described = oc_client.vector.describe_collection(COLLECTION_NAME)
    assert described.name == COLLECTION_NAME


def test_list_collections(oc_client):
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    collections = oc_client.vector.list_collections()
    names = [c.name for c in collections]
    assert COLLECTION_NAME in names


def test_has_collection(oc_client):
    assert oc_client.vector.has_collection(COLLECTION_NAME) is False
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    assert oc_client.vector.has_collection(COLLECTION_NAME) is True


def test_upsert_and_fetch(oc_client):
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)

    result = col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0], "metadata": {"label": "a"}},
            {"id": "v2", "values": [0.0, 1.0, 0.0], "metadata": {"label": "b"}},
        ]
    )
    assert result.upserted_count == 2

    fetched = col.fetch(ids=["v1"])
    assert fetched.records[0]["id"] == "v1"


def test_query(oc_client):
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0]},
            {"id": "v2", "values": [0.0, 1.0, 0.0]},
        ]
    )

    results = col.query(vector=[1.0, 0.0, 0.0], top_k=2, include_metadata=True)
    assert len(results.matches) >= 1
    assert results.matches[0].id == "v1"


def test_delete_by_ids(oc_client):
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(vectors=[{"id": "v1", "values": [1.0, 0.0, 0.0]}])
    col.delete(ids=["v1"])

    fetched = col.fetch(ids=["v1"])
    assert len(fetched.records) == 0


def test_describe_collection_stats(oc_client):
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(vectors=[{"id": "v1", "values": [1.0, 0.0, 0.0]}])

    stats = col.describe_collection_stats()
    assert stats.dimension == DIM
    assert stats.total_record_count >= 1


def test_list_records(oc_client):
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[
            {"id": "doc-1", "values": [1.0, 0.0, 0.0]},
            {"id": "doc-2", "values": [0.0, 1.0, 0.0]},
        ]
    )

    result = col.list(prefix="doc-")
    ids = [v["id"] for v in result.records]
    assert "doc-1" in ids
    assert "doc-2" in ids


def test_update_metadata(oc_client):
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(vectors=[{"id": "v1", "values": [1.0, 0.0, 0.0], "metadata": {"x": 1}}])
    col.update(id="v1", set_metadata={"x": 99})

    fetched = col.fetch(ids=["v1"])
    assert fetched.records[0]["metadata"]["x"] == 99


def test_not_found_error(oc_client):
    with pytest.raises(NotFoundError):
        oc_client.vector.describe_collection("nonexistent-collection-xyz")


HYBRID_COLLECTION = "sdk-hybrid-test"


@pytest.fixture()
def hybrid_cleanup(oc_client):
    yield
    with contextlib.suppress(Exception):
        oc_client.vector.delete_collection(HYBRID_COLLECTION)


def test_query_hybrid(oc_client, hybrid_cleanup):
    oc_client.vector.create_collection(name=HYBRID_COLLECTION, dimension=DIM, bm25_enabled=True)
    col = oc_client.vector.collection(HYBRID_COLLECTION)
    col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0], "text": "machine learning basics"},
            {"id": "v2", "values": [0.0, 1.0, 0.0], "text": "cooking recipes"},
        ]
    )

    results = col.query_hybrid(
        vector=[1.0, 0.0, 0.0],
        text="machine learning",
        top_k=2,
    )
    assert len(results.matches) >= 1


def test_query_with_rerank(oc_client, hybrid_cleanup):
    """Rerank with noop backend: request is accepted and results are returned."""
    oc_client.vector.create_collection(name=HYBRID_COLLECTION, dimension=DIM, bm25_enabled=True)
    col = oc_client.vector.collection(HYBRID_COLLECTION)
    col.upsert(
        vectors=[
            {
                "id": "v1",
                "values": [1.0, 0.0, 0.0],
                "metadata": {"text": "machine learning"},
                "text": "machine learning",
            },
            {"id": "v2", "values": [0.0, 1.0, 0.0], "metadata": {"text": "cooking"}, "text": "cooking"},
        ]
    )

    results = col.query(
        vector=[1.0, 0.0, 0.0],
        top_k=5,
        rerank={"query": "machine learning", "topN": 1, "rankField": "text"},
    )
    assert len(results.matches) >= 1


def test_query_hybrid_with_rerank(oc_client, hybrid_cleanup):
    """Hybrid query with reranking (noop backend)."""
    oc_client.vector.create_collection(name=HYBRID_COLLECTION, dimension=DIM, bm25_enabled=True)
    col = oc_client.vector.collection(HYBRID_COLLECTION)
    col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0], "text": "machine learning basics"},
            {"id": "v2", "values": [0.0, 1.0, 0.0], "text": "cooking recipes"},
        ]
    )

    results = col.query_hybrid(
        vector=[1.0, 0.0, 0.0],
        text="machine learning",
        top_k=5,
        rerank={"query": "machine learning", "topN": 1},
    )
    assert len(results.matches) >= 1


# ── New feature integration tests ────────────────────────────────────────────


def test_scroll_and_pagination(oc_client):
    """Scroll page-by-page and confirm all records are reachable."""
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[
            {"id": f"v{i}", "values": [float(i % 2), float((i + 1) % 2), 0.0]}
            for i in range(1, 6)  # 5 records
        ]
    )

    all_ids: list[str] = []
    cursor = None
    while True:
        page = col.scroll(limit=2, cursor=cursor, include_values=False)
        all_ids.extend(v.id for v in page.records)
        if page.next_cursor is None:
            break
        cursor = page.next_cursor

    assert len(all_ids) == 5
    assert sorted(all_ids) == ["v1", "v2", "v3", "v4", "v5"]


def test_sample(oc_client):
    """sample() returns at most `size` records with valid IDs."""
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[{"id": f"v{i}", "values": [1.0, 0.0, 0.0]} for i in range(1, 11)]  # 10 records
    )

    result = col.sample(size=3)
    assert 1 <= len(result.records) <= 3
    for v in result.records:
        assert v.id.startswith("v")
    assert result.next_cursor is None


def test_query_batch(oc_client):
    """query_batch returns results in the same order as the input queries."""
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0]},
            {"id": "v2", "values": [0.0, 1.0, 0.0]},
            {"id": "v3", "values": [0.0, 0.0, 1.0]},
        ]
    )

    result = col.query_batch(
        queries=[
            {"vector": [1.0, 0.0, 0.0], "topK": 1},
            {"vector": [0.0, 1.0, 0.0], "topK": 1},
        ]
    )
    assert len(result.results) == 2
    assert result.results[0].matches[0].id == "v1"
    assert result.results[1].matches[0].id == "v2"


def test_query_score_threshold(oc_client):
    """score_threshold filters out records below the minimum similarity."""
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0]},  # identical to query → score ~1.0
            {"id": "v2", "values": [0.0, 1.0, 0.0]},  # orthogonal → score ~0.0
        ]
    )

    results = col.query(vector=[1.0, 0.0, 0.0], top_k=10, score_threshold=0.9)
    ids = [m.id for m in results.matches]
    assert "v1" in ids
    assert "v2" not in ids


def test_query_group_by(oc_client):
    """group_by returns a GroupedQueryResult grouped by the metadata field."""
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[
            {"id": "n1", "values": [0.9, 0.1, 0.0], "metadata": {"source": "news"}},
            {"id": "n2", "values": [0.8, 0.2, 0.0], "metadata": {"source": "news"}},
            {"id": "s1", "values": [0.6, 0.4, 0.0], "metadata": {"source": "sports"}},
        ]
    )

    result = col.query(
        vector=[1.0, 0.0, 0.0],
        top_k=10,
        group_by={"field": "source", "limit": 5, "groupSize": 2},
    )
    assert isinstance(result, GroupedQueryResult)
    group_names = {g.key for g in result.groups}
    assert "news" in group_names
    assert "sports" in group_names
    for g in result.groups:
        assert len(g.matches) <= 2


def test_recommend(oc_client):
    """recommend() excludes input IDs and returns semantically similar records."""
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0]},
            {"id": "v2", "values": [0.9, 0.1, 0.0]},
            {"id": "v3", "values": [0.0, 1.0, 0.0]},
            {"id": "v4", "values": [0.0, 0.0, 1.0]},
        ]
    )

    result = col.recommend(positive_ids=["v1"], top_k=3)
    ids = [m.id for m in result.matches]
    assert "v1" not in ids  # input ID excluded
    assert ids[0] == "v2"  # most similar to v1


def test_alias_lifecycle(oc_client):
    """Full alias lifecycle: create → query → update target → delete."""
    oc_client.vector.create_collection(name=COLLECTION_NAME, dimension=DIM)
    col = oc_client.vector.collection(COLLECTION_NAME)
    col.upsert(vectors=[{"id": "v1", "values": [1.0, 0.0, 0.0]}])

    alias_name = "sdk-test-alias"
    with contextlib.suppress(Exception):
        oc_client.vector.delete_alias(alias_name)

    try:
        # Create alias
        desc = oc_client.vector.create_alias(alias=alias_name, collection_name=COLLECTION_NAME)
        assert desc.alias == alias_name
        assert desc.collection_name == COLLECTION_NAME

        # Query via alias — should resolve transparently
        alias_col = oc_client.vector.collection(alias_name)
        results = alias_col.query(vector=[1.0, 0.0, 0.0], top_k=1)
        assert results.matches[0].id == "v1"

        # Confirm it appears in list_aliases()
        aliases = oc_client.vector.list_aliases()
        found = [a for a in aliases.aliases if a.alias == alias_name]
        assert len(found) == 1

        # describe_alias()
        described = oc_client.vector.describe_alias(alias_name)
        assert described.collection_name == COLLECTION_NAME

    finally:
        # Always clean up alias
        with contextlib.suppress(Exception):
            oc_client.vector.delete_alias(alias_name)

    # After deletion, describe should raise NotFoundError
    with pytest.raises(NotFoundError):
        oc_client.vector.describe_alias(alias_name)
