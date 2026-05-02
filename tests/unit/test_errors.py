"""Unit tests for the v0.2.0 error contract: typed exception subclasses,
structured `details`, `request_id` capture, and refined retry policy.
"""

import httpx
import pytest
import respx

from onecortex import (
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    FacetFieldInvalidError,
    FilterMalformedError,
    FilterUnsupportedOperatorError,
    GroupByFieldMissingError,
    HybridRequiresBm25Error,
    IndexNotReadyError,
    Onecortex,
    OnecortexServerError,
    RerankerConfigError,
    RerankerRateLimitedError,
    RerankerTimeoutError,
    RerankerUpstreamError,
    SparseNotSupportedError,
)

BASE = "http://test-server:8080"
VP = "/vector/v1"
COL_NAME = "test-col"
COL_BASE = f"{BASE}{VP}/collections/{COL_NAME}"


def make_collection():
    pc = Onecortex(url=BASE)
    return pc.vector.collection(COL_NAME)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Skip retry backoff sleeps in unit tests."""
    monkeypatch.setattr("onecortex._http.time.sleep", lambda *_a, **_k: None)


def _err(code: str, message: str = "boom", details: dict | None = None) -> dict:
    body: dict = {"error": {"code": code, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body


# Each row: (code, status, exc_class, details, target_path, method)
# All hit the upsert path except for the few cases that need a different endpoint.
TYPED_CASES = [
    (
        "DIMENSION_MISMATCH",
        400,
        DimensionMismatchError,
        {"recordId": "r1", "expected": 3, "got": 4},
    ),
    (
        "SPARSE_NOT_SUPPORTED",
        400,
        SparseNotSupportedError,
        {"recordId": "r1"},
    ),
    (
        "FILTER_MALFORMED",
        400,
        FilterMalformedError,
        {"reason": "unterminated string"},
    ),
    (
        "FILTER_UNSUPPORTED_OPERATOR",
        400,
        FilterUnsupportedOperatorError,
        {"operator": "$nope"},
    ),
    (
        "HYBRID_REQUIRES_BM25",
        400,
        HybridRequiresBm25Error,
        {"collection": COL_NAME},
    ),
    (
        "GROUPBY_FIELD_MISSING",
        400,
        GroupByFieldMissingError,
        {"field": "category"},
    ),
    (
        "FACET_FIELD_INVALID",
        400,
        FacetFieldInvalidError,
        {"field": "x", "reason": "not faceted"},
    ),
    (
        "COLLECTION_NOT_FOUND",
        404,
        CollectionNotFoundError,
        {"collection": COL_NAME},
    ),
    (
        "COLLECTION_ALREADY_EXISTS",
        409,
        CollectionAlreadyExistsError,
        {"collection": COL_NAME},
    ),
    (
        "INDEX_NOT_READY",
        409,
        IndexNotReadyError,
        {"collection": COL_NAME, "status": "indexing"},
    ),
    (
        "RERANKER_RATE_LIMITED",
        429,
        RerankerRateLimitedError,
        {"retries": 3},
    ),
    (
        "RERANKER_UPSTREAM",
        502,
        RerankerUpstreamError,
        {"kind": "status", "upstreamStatus": 500},
    ),
    (
        "RERANKER_CONFIG",
        503,
        RerankerConfigError,
        {},
    ),
    (
        "RERANKER_TIMEOUT",
        504,
        RerankerTimeoutError,
        {"kind": "timeout"},
    ),
]


@pytest.mark.parametrize("code,status,exc_class,details", TYPED_CASES)
@respx.mock
def test_typed_exception_per_code(code, status, exc_class, details):
    body = _err(code, "msg", {**details, "requestId": "req-abc-123"})
    # 429/504 retry — return the same error 4 times so we exhaust retries
    # and surface the typed exception. 502/503 should NOT retry, but mocking
    # them once is enough; mocking 4x is harmless.
    respx.post(f"{COL_BASE}/records/upsert").mock(side_effect=[httpx.Response(status, json=body)] * 4)
    col = make_collection()
    with pytest.raises(exc_class) as exc_info:
        col.upsert([{"id": "r1", "values": [1.0, 2.0, 3.0]}])

    err = exc_info.value
    assert err.code == code
    assert err.status_code == status
    assert err.request_id == "req-abc-123"
    for k, v in details.items():
        assert err.details.get(k) == v


@respx.mock
def test_request_id_falls_back_to_header():
    """If the body has no requestId but the X-Request-Id header is set, use it."""
    respx.post(f"{COL_BASE}/records/upsert").mock(
        return_value=httpx.Response(
            400,
            json=_err("DIMENSION_MISMATCH", "bad dim", {"expected": 3, "got": 5}),
            headers={"x-request-id": "header-rid-7"},
        )
    )
    col = make_collection()
    with pytest.raises(DimensionMismatchError) as exc_info:
        col.upsert([{"id": "r1", "values": [1.0, 2.0, 3.0, 4.0, 5.0]}])
    assert exc_info.value.request_id == "header-rid-7"


@respx.mock
def test_504_retries_then_succeeds():
    """504 RERANKER_TIMEOUT is retryable — first attempt fails, second succeeds."""
    respx.post(f"{COL_BASE}/records/upsert").mock(
        side_effect=[
            httpx.Response(504, json=_err("RERANKER_TIMEOUT", "slow", {"kind": "timeout"})),
            httpx.Response(200, json={"upsertedCount": 1}),
        ]
    )
    col = make_collection()
    result = col.upsert([{"id": "r1", "values": [1.0, 2.0, 3.0]}])
    assert result.upserted_count == 1


@respx.mock
def test_502_does_not_retry():
    """502 RERANKER_UPSTREAM must not be retried — single attempt, immediate raise."""
    route = respx.post(f"{COL_BASE}/records/upsert").mock(
        return_value=httpx.Response(502, json=_err("RERANKER_UPSTREAM", "upstream", {"kind": "status"}))
    )
    col = make_collection()
    with pytest.raises(RerankerUpstreamError):
        col.upsert([{"id": "r1", "values": [1.0, 2.0, 3.0]}])
    assert route.call_count == 1


@respx.mock
def test_503_does_not_retry():
    """503 RERANKER_CONFIG must not be retried — single attempt, immediate raise."""
    route = respx.post(f"{COL_BASE}/records/upsert").mock(
        return_value=httpx.Response(503, json=_err("RERANKER_CONFIG", "no key"))
    )
    col = make_collection()
    with pytest.raises(RerankerConfigError):
        col.upsert([{"id": "r1", "values": [1.0, 2.0, 3.0]}])
    assert route.call_count == 1


@respx.mock
def test_500_retries():
    """500 INTERNAL retries (transient server error)."""
    respx.post(f"{COL_BASE}/records/upsert").mock(
        side_effect=[
            httpx.Response(500, json=_err("INTERNAL", "oops")),
            httpx.Response(200, json={"upsertedCount": 1}),
        ]
    )
    col = make_collection()
    result = col.upsert([{"id": "r1", "values": [1.0, 2.0, 3.0]}])
    assert result.upserted_count == 1


@respx.mock
def test_unknown_code_falls_back_to_server_error():
    """Codes not in the map surface as OnecortexServerError but still carry code/details."""
    respx.post(f"{COL_BASE}/records/upsert").mock(
        return_value=httpx.Response(418, json=_err("TEAPOT", "i am a teapot", {"requestId": "rid-9"}))
    )
    col = make_collection()
    with pytest.raises(OnecortexServerError) as exc_info:
        col.upsert([{"id": "r1", "values": [1.0, 2.0, 3.0]}])
    err = exc_info.value
    assert err.code == "TEAPOT"
    assert err.request_id == "rid-9"
