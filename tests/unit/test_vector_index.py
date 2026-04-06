import httpx
import pytest
import respx

from onecortex import Onecortex

BASE = "http://test-server:8080"
VP = "/v1/vector"
COL_NAME = "test-col"
COL_BASE = f"{BASE}{VP}/collections/{COL_NAME}"

QUERY_RESPONSE = {
    "matches": [{"id": "v1", "score": 0.99}],
    "namespace": "",
    "results": [],
}

UPSERT_RESPONSE = {"upsertedCount": 2}

FETCH_RESPONSE = {
    "records": {
        "v1": {"id": "v1", "values": [1.0, 0.0, 0.0], "metadata": {}},
    },
    "namespace": "",
}

LIST_RESPONSE = {
    "records": [{"id": "v1"}, {"id": "v2"}],
    "namespace": "",
}

STATS_RESPONSE = {
    "namespaces": {"": {"recordCount": 2}},
    "dimension": 3,
    "collectionFullness": 0.001,
    "totalRecordCount": 2,
}


def make_collection():
    pc = Onecortex(url=BASE, api_key="key123")
    return pc.vector.collection(COL_NAME)


@respx.mock
def test_upsert():
    respx.post(f"{COL_BASE}/records/upsert").mock(return_value=httpx.Response(200, json=UPSERT_RESPONSE))
    col = make_collection()
    result = col.upsert(
        vectors=[
            {"id": "v1", "values": [1.0, 0.0, 0.0]},
            {"id": "v2", "values": [0.0, 1.0, 0.0]},
        ]
    )
    assert result.upserted_count == 2


@respx.mock
def test_upsert_batch():
    respx.post(f"{COL_BASE}/records/upsert").mock(return_value=httpx.Response(200, json={"upsertedCount": 1}))
    col = make_collection()
    # 3 vectors with batch_size=2 → 2 requests
    total = col.upsert_batch(
        vectors=[{"id": f"v{i}", "values": [float(i), 0.0, 0.0]} for i in range(3)],
        batch_size=2,
    )
    assert total == 2  # 2 batches x 1 upsertedCount each


@respx.mock
def test_query():
    respx.post(f"{COL_BASE}/query").mock(return_value=httpx.Response(200, json=QUERY_RESPONSE))
    col = make_collection()
    result = col.query(vector=[1.0, 0.0, 0.0], top_k=1)
    assert result.matches[0].id == "v1"
    assert result.matches[0].score == 0.99


@respx.mock
def test_query_by_id():
    respx.post(f"{COL_BASE}/query").mock(return_value=httpx.Response(200, json=QUERY_RESPONSE))
    col = make_collection()
    result = col.query(vector=[], id="v1", top_k=1)
    assert result.matches[0].id == "v1"


@respx.mock
def test_fetch():
    respx.post(f"{COL_BASE}/records/fetch").mock(return_value=httpx.Response(200, json=FETCH_RESPONSE))
    col = make_collection()
    result = col.fetch(ids=["v1"])
    assert "v1" in result.records


@respx.mock
def test_delete_by_ids():
    respx.post(f"{COL_BASE}/records/delete").mock(return_value=httpx.Response(200, json={}))
    col = make_collection()
    col.delete(ids=["v1"])  # should not raise


@respx.mock
def test_delete_all():
    respx.post(f"{COL_BASE}/records/delete").mock(return_value=httpx.Response(200, json={}))
    col = make_collection()
    col.delete(delete_all=True)


def test_delete_no_args_raises():
    col = make_collection()
    with pytest.raises(ValueError):
        col.delete()


@respx.mock
def test_update():
    respx.post(f"{COL_BASE}/records/update").mock(return_value=httpx.Response(200, json={}))
    col = make_collection()
    col.update(id="v1", set_metadata={"key": "new"})


@respx.mock
def test_list():
    respx.get(f"{COL_BASE}/records/list").mock(return_value=httpx.Response(200, json=LIST_RESPONSE))
    col = make_collection()
    result = col.list()
    assert len(result.records) == 2


@respx.mock
def test_describe_collection_stats():
    respx.post(f"{COL_BASE}/describe_collection_stats").mock(return_value=httpx.Response(200, json=STATS_RESPONSE))
    col = make_collection()
    stats = col.describe_collection_stats()
    assert stats.total_record_count == 2
    assert stats.dimension == 3


@respx.mock
def test_query_with_rerank():
    route = respx.post(f"{COL_BASE}/query").mock(return_value=httpx.Response(200, json=QUERY_RESPONSE))
    col = make_collection()
    result = col.query(
        vector=[1.0, 0.0, 0.0],
        top_k=5,
        rerank={"query": "machine learning", "topN": 2, "rankField": "text"},
    )
    assert result.matches[0].id == "v1"
    # Verify the rerank field was sent in the request body
    sent_body = route.calls[0].request.content
    import json

    body = json.loads(sent_body)
    assert body["rerank"] == {"query": "machine learning", "topN": 2, "rankField": "text"}


@respx.mock
def test_query_hybrid():
    respx.post(f"{COL_BASE}/query/hybrid").mock(return_value=httpx.Response(200, json=QUERY_RESPONSE))
    col = make_collection()
    result = col.query_hybrid(vector=[1.0, 0.0, 0.0], text="hello", top_k=5)
    assert result.matches[0].id == "v1"


@respx.mock
def test_query_hybrid_with_rerank():
    route = respx.post(f"{COL_BASE}/query/hybrid").mock(return_value=httpx.Response(200, json=QUERY_RESPONSE))
    col = make_collection()
    result = col.query_hybrid(
        vector=[1.0, 0.0, 0.0],
        text="hello",
        top_k=5,
        rerank={"query": "hello world", "topN": 3},
    )
    assert result.matches[0].id == "v1"
    import json

    body = json.loads(route.calls[0].request.content)
    assert body["rerank"] == {"query": "hello world", "topN": 3}


# ── New feature tests ────────────────────────────────────────────────────────

SCROLL_RESPONSE = {
    "records": [
        {"id": "v1", "values": [1.0, 0.0, 0.0]},
        {"id": "v2", "values": [0.0, 1.0, 0.0]},
    ],
    "namespace": "",
    "nextCursor": "cursor-xyz",
}

SCROLL_LAST_PAGE_RESPONSE = {
    "records": [{"id": "v3"}],
    "namespace": "",
    # no nextCursor
}

BATCH_RESPONSE = {
    "results": [
        {"matches": [{"id": "v1", "score": 0.9}], "namespace": ""},
        {"matches": [{"id": "v2", "score": 0.8}], "namespace": ""},
    ]
}

GROUP_RESPONSE = {
    "matches": [
        {"group": "doc-1", "matches": [{"id": "c1", "score": 0.9}, {"id": "c2", "score": 0.8}]},
        {"group": "doc-2", "matches": [{"id": "c3", "score": 0.7}]},
    ],
    "namespace": "",
}

RECOMMEND_RESPONSE = {
    "matches": [{"id": "v2", "score": 0.88}, {"id": "v3", "score": 0.72}],
    "namespace": "",
}


@respx.mock
def test_scroll_basic():
    respx.post(f"{COL_BASE}/records/scroll").mock(return_value=httpx.Response(200, json=SCROLL_RESPONSE))
    col = make_collection()
    result = col.scroll(limit=2, include_values=True)
    assert len(result.records) == 2
    assert result.records[0].id == "v1"
    assert result.next_cursor == "cursor-xyz"


@respx.mock
def test_scroll_no_cursor_on_last_page():
    respx.post(f"{COL_BASE}/records/scroll").mock(return_value=httpx.Response(200, json=SCROLL_LAST_PAGE_RESPONSE))
    col = make_collection()
    result = col.scroll(cursor="cursor-xyz")
    assert len(result.records) == 1
    assert result.next_cursor is None


@respx.mock
def test_sample_basic():
    respx.post(f"{COL_BASE}/sample").mock(return_value=httpx.Response(200, json=SCROLL_LAST_PAGE_RESPONSE))
    col = make_collection()
    result = col.sample(size=5)
    assert len(result.records) == 1
    assert result.next_cursor is None  # sample never has a next cursor


@respx.mock
def test_query_batch_success():
    import json

    route = respx.post(f"{COL_BASE}/query/batch").mock(return_value=httpx.Response(200, json=BATCH_RESPONSE))
    col = make_collection()
    result = col.query_batch(
        queries=[
            {"vector": [1.0, 0.0, 0.0], "topK": 1},
            {"vector": [0.0, 1.0, 0.0], "topK": 1},
        ]
    )
    assert len(result.results) == 2
    assert result.results[0].matches[0].id == "v1"
    assert result.results[1].matches[0].id == "v2"
    body = json.loads(route.calls[0].request.content)
    assert len(body["queries"]) == 2


def test_query_batch_empty_raises():
    col = make_collection()
    with pytest.raises(ValueError, match="empty"):
        col.query_batch(queries=[])


def test_query_batch_too_many_raises():
    col = make_collection()
    with pytest.raises(ValueError, match="10"):
        col.query_batch(queries=[{"vector": [1.0, 0.0, 0.0], "topK": 1}] * 11)


@respx.mock
def test_query_with_score_threshold():
    import json

    route = respx.post(f"{COL_BASE}/query").mock(return_value=httpx.Response(200, json=QUERY_RESPONSE))
    col = make_collection()
    result = col.query(vector=[1.0, 0.0, 0.0], top_k=5, score_threshold=0.8)
    assert result.matches[0].id == "v1"
    body = json.loads(route.calls[0].request.content)
    assert body["scoreThreshold"] == 0.8


@respx.mock
def test_query_with_group_by():
    import json

    route = respx.post(f"{COL_BASE}/query").mock(return_value=httpx.Response(200, json=GROUP_RESPONSE))
    col = make_collection()
    group_by = {"field": "doc", "limit": 5, "groupSize": 2}
    result = col.query(vector=[1.0, 0.0, 0.0], top_k=20, group_by=group_by)
    # Returns GroupedQueryResult, not QueryResult
    from onecortex.vector.models import GroupedQueryResult

    assert isinstance(result, GroupedQueryResult)
    assert len(result.groups) == 2
    assert result.groups[0].group == "doc-1"
    assert result.groups[0].matches[0].id == "c1"
    body = json.loads(route.calls[0].request.content)
    assert body["groupBy"] == group_by
    assert body["includeMetadata"] is True  # forced on by group_by


@respx.mock
def test_recommend_basic():
    import json

    route = respx.post(f"{COL_BASE}/recommend").mock(return_value=httpx.Response(200, json=RECOMMEND_RESPONSE))
    col = make_collection()
    result = col.recommend(positive_ids=["v1"], top_k=2)
    from onecortex.vector.models import RecommendResult

    assert isinstance(result, RecommendResult)
    assert result.matches[0].id == "v2"
    body = json.loads(route.calls[0].request.content)
    assert body["positiveIds"] == ["v1"]
    assert body["negativeIds"] == []
    assert body["topK"] == 2


@respx.mock
def test_recommend_with_negatives():
    import json

    route = respx.post(f"{COL_BASE}/recommend").mock(return_value=httpx.Response(200, json=RECOMMEND_RESPONSE))
    col = make_collection()
    col.recommend(positive_ids=["v1"], negative_ids=["v9"], top_k=2, score_threshold=0.5)
    body = json.loads(route.calls[0].request.content)
    assert body["negativeIds"] == ["v9"]
    assert body["scoreThreshold"] == 0.5
