"""Integration tests for the v0.2.0 typed error contract.

Requires a live Onecortex cluster (postgres + auth + vector + apisix) and a
valid bearer token in ONECORTEX_ACCESS_TOKEN (or ONECORTEX_EMAIL/PASSWORD).

Reranker codes (RERANKER_RATE_LIMITED, RERANKER_TIMEOUT, RERANKER_UPSTREAM,
RERANKER_CONFIG) are not exercised here — they require a controllable
upstream reranker. Unit tests cover them.
"""

import contextlib
import uuid

import pytest

from onecortex import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    GroupByFieldMissingError,
    HybridRequiresBm25Error,
    NotFoundError,
    SparseNotSupportedError,
)

DIM = 3


def _name(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.fixture
def fresh_collection(oc_client):
    """Create a unique cosine collection, yield its name, clean up after."""
    name = _name("err-int")
    oc_client.vector.create_collection(name=name, dimension=DIM, metric="cosine")
    yield name
    with contextlib.suppress(Exception):
        oc_client.vector.delete_collection(name)


def test_sparse_not_supported(oc_client, fresh_collection):
    col = oc_client.vector.collection(fresh_collection)
    with pytest.raises(SparseNotSupportedError) as exc_info:
        col.upsert(
            [
                {
                    "id": "r1",
                    "values": [1.0, 0.0, 0.0],
                    "sparseValues": {"indices": [0, 1], "values": [0.1, 0.2]},
                }
            ]
        )
    err = exc_info.value
    assert err.status_code == 400
    assert err.code == "SPARSE_NOT_SUPPORTED"
    assert err.details.get("recordId") == "r1"
    assert err.request_id


def test_dimension_mismatch_on_upsert(oc_client, fresh_collection):
    col = oc_client.vector.collection(fresh_collection)
    with pytest.raises(DimensionMismatchError) as exc_info:
        col.upsert([{"id": "r1", "values": [1.0, 0.0, 0.0, 0.0, 0.0]}])
    err = exc_info.value
    assert err.status_code == 400
    assert err.details.get("expected") == DIM
    assert err.details.get("got") == 5
    assert err.request_id


def test_dimension_mismatch_on_query(oc_client, fresh_collection):
    col = oc_client.vector.collection(fresh_collection)
    col.upsert([{"id": "r1", "values": [1.0, 0.0, 0.0]}])
    with pytest.raises(DimensionMismatchError) as exc_info:
        col.query(vector=[1.0, 0.0], top_k=1)
    assert exc_info.value.details.get("expected") == DIM
    assert exc_info.value.details.get("got") == 2


def test_collection_not_found(oc_client):
    missing = _name("missing")
    with pytest.raises(CollectionNotFoundError) as exc_info:
        oc_client.vector.collection(missing).query(vector=[1.0, 0.0, 0.0], top_k=1)
    err = exc_info.value
    assert err.status_code == 404
    assert err.code == "COLLECTION_NOT_FOUND"
    assert err.details.get("collection") == missing
    assert err.request_id


def test_collection_already_exists(oc_client, fresh_collection):
    with pytest.raises(CollectionAlreadyExistsError) as exc_info:
        oc_client.vector.create_collection(name=fresh_collection, dimension=DIM, metric="cosine")
    err = exc_info.value
    assert err.status_code == 409
    assert err.code == "COLLECTION_ALREADY_EXISTS"
    assert err.details.get("collection") == fresh_collection


def test_hybrid_requires_bm25(oc_client):
    """A non-BM25 cosine collection cannot serve a hybrid query.

    Since onecortex-vector v0.3 the server defaults bm25Enabled to true on
    new collections, so this test must opt out explicitly.
    """
    name = _name("err-int-nobm25")
    oc_client.vector.create_collection(name=name, dimension=DIM, metric="cosine", bm25_enabled=False)
    try:
        col = oc_client.vector.collection(name)
        col.upsert([{"id": "r1", "values": [1.0, 0.0, 0.0], "text": "hello"}])
        with pytest.raises(HybridRequiresBm25Error) as exc_info:
            col.query_hybrid(vector=[1.0, 0.0, 0.0], text="hello", top_k=1)
        err = exc_info.value
        assert err.status_code == 400
        assert err.code == "HYBRID_REQUIRES_BM25"
        assert err.details.get("collection") == name
    finally:
        with contextlib.suppress(Exception):
            oc_client.vector.delete_collection(name)


def test_group_by_field_missing(oc_client, fresh_collection):
    """groupBy on a field that is absent on every record returns 400."""
    col = oc_client.vector.collection(fresh_collection)
    col.upsert([{"id": "r1", "values": [1.0, 0.0, 0.0]}])
    with pytest.raises(GroupByFieldMissingError) as exc_info:
        col.query(
            vector=[1.0, 0.0, 0.0],
            top_k=5,
            group_by={"field": "nonexistent_field", "groupSize": 2},
        )
    err = exc_info.value
    assert err.status_code == 400
    assert err.details.get("field") == "nonexistent_field"


def test_request_id_present_on_every_error(oc_client):
    """Every error must carry a request_id (from body details or X-Request-Id header).

    describe_collection (control plane) emits the legacy NOT_FOUND code rather
    than the typed COLLECTION_NOT_FOUND, so we catch the parent NotFoundError.
    """
    with pytest.raises(NotFoundError) as exc_info:
        oc_client.vector.describe_collection(_name("absent"))
    rid = exc_info.value.request_id
    assert isinstance(rid, str) and len(rid) > 0
