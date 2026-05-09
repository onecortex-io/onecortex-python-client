import json

import httpx
import pytest
import respx

from onecortex import EmbedderSpec, Onecortex
from onecortex.exceptions import NotFoundError

BASE = "http://test-server:8080"
VP = "/vector/v1"

COLLECTION_RESPONSE = {
    "name": "test-col",
    "dimension": 3,
    "metric": "cosine",
    "status": {"ready": True, "state": "Ready"},
    "host": "test-server:8080",
    "vectorType": "dense",
    "bm25Enabled": False,
    "deletionProtected": None,
}


@respx.mock
def test_create_collection():
    respx.post(f"{BASE}{VP}/collections").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE)
    col = pc.vector.create_collection(name="test-col", dimension=3, metric="cosine")
    assert col.name == "test-col"
    assert col.dimension == 3


@respx.mock
def test_create_collection_ignores_spec():
    respx.post(f"{BASE}{VP}/collections").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE)
    # spec= is an unknown arg — must not raise
    col = pc.vector.create_collection(
        name="test-col",
        dimension=3,
        spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
    )
    assert col.name == "test-col"


@respx.mock
def test_describe_collection():
    respx.get(f"{BASE}{VP}/collections/test-col").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE)
    col = pc.vector.describe_collection("test-col")
    assert col.metric == "cosine"


@respx.mock
def test_list_collections():
    respx.get(f"{BASE}{VP}/collections").mock(
        return_value=httpx.Response(200, json={"collections": [COLLECTION_RESPONSE]})
    )
    pc = Onecortex(url=BASE)
    collections = pc.vector.list_collections()
    assert len(collections) == 1
    assert collections[0].name == "test-col"


@respx.mock
def test_delete_collection():
    respx.delete(f"{BASE}{VP}/collections/test-col").mock(return_value=httpx.Response(202))
    pc = Onecortex(url=BASE)
    pc.vector.delete_collection("test-col")  # should not raise


@respx.mock
def test_has_collection_true():
    respx.get(f"{BASE}{VP}/collections/test-col").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE)
    assert pc.vector.has_collection("test-col") is True


@respx.mock
def test_has_collection_false():
    respx.get(f"{BASE}{VP}/collections/missing").mock(
        return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "not found"}})
    )
    pc = Onecortex(url=BASE)
    assert pc.vector.has_collection("missing") is False


@respx.mock
def test_configure_collection():
    respx.patch(f"{BASE}{VP}/collections/test-col").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE)
    result = pc.vector.configure_collection("test-col", tags={"env": "prod"})
    assert result.name == "test-col"


@respx.mock
def test_configure_collection_forwards_bm25_enabled():
    """Regression: bm25_enabled was silently swallowed by **kwargs."""
    captured: dict[str, str] = {}

    def _capture(request: httpx.Request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json={**COLLECTION_RESPONSE, "bm25Enabled": True})

    respx.patch(f"{BASE}{VP}/collections/test-col").mock(side_effect=_capture)
    pc = Onecortex(url=BASE)
    result = pc.vector.configure_collection("test-col", bm25_enabled=True)
    body = captured["body"]
    assert '"bm25Enabled": true' in body or '"bm25Enabled":true' in body
    assert result.bm25_enabled is True


def test_configure_collection_requires_at_least_one_field():
    pc = Onecortex(url=BASE)
    with pytest.raises(ValueError, match="at least one"):
        pc.vector.configure_collection("test-col")


def test_configure_collection_rejects_unknown_kwargs():
    """Removing **kwargs means typos must fail loudly."""
    pc = Onecortex(url=BASE)
    with pytest.raises(TypeError):
        pc.vector.configure_collection("test-col", bm25=True)  # type: ignore[call-arg]


@respx.mock
def test_not_found_raises():
    respx.get(f"{BASE}{VP}/collections/missing").mock(
        return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "not found"}})
    )
    pc = Onecortex(url=BASE)
    with pytest.raises(NotFoundError):
        pc.vector.describe_collection("missing")


# ── Alias tests ──────────────────────────────────────────────────────────────

ALIAS_RESPONSE = {"alias": "prod", "collectionName": "my-col-v2"}
ALIAS_LIST_RESPONSE = {
    "aliases": [
        {"alias": "prod", "collectionName": "my-col-v2"},
        {"alias": "staging", "collectionName": "my-col-v1"},
    ]
}


@respx.mock
def test_create_collection_with_embedder_dict():
    captured: dict[str, str] = {}

    def _capture(request: httpx.Request):
        captured["body"] = request.read().decode()
        return httpx.Response(
            200,
            json={
                **COLLECTION_RESPONSE,
                "embedder": {
                    "backend": "openai",
                    "model": "text-embedding-3-small",
                    "inputType": "document",
                },
            },
        )

    respx.post(f"{BASE}{VP}/collections").mock(side_effect=_capture)
    pc = Onecortex(url=BASE)
    col = pc.vector.create_collection(
        name="test-col",
        dimension=3,
        embedder={
            "backend": "openai",
            "model": "text-embedding-3-small",
            "inputType": "document",
        },
    )
    body = json.loads(captured["body"])
    assert body["embedder"] == {
        "backend": "openai",
        "model": "text-embedding-3-small",
        "inputType": "document",
    }
    assert col.embedder is not None
    assert col.embedder.backend == "openai"
    assert col.embedder.input_type == "document"


@respx.mock
def test_create_collection_with_embedder_spec_omits_none_alias():
    captured: dict[str, str] = {}

    def _capture(request: httpx.Request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=COLLECTION_RESPONSE)

    respx.post(f"{BASE}{VP}/collections").mock(side_effect=_capture)
    pc = Onecortex(url=BASE)
    pc.vector.create_collection(
        name="test-col",
        dimension=3,
        embedder=EmbedderSpec(backend="voyage", model="voyage-3"),
    )
    body = json.loads(captured["body"])
    assert body["embedder"] == {"backend": "voyage", "model": "voyage-3"}
    assert "inputType" not in body["embedder"]


@respx.mock
def test_create_collection_bm25_default_omitted_when_unset():
    """When bm25_enabled is not specified, the SDK does not send the field
    so the server-side default (true since v0.3) takes effect."""
    captured: dict[str, str] = {}

    def _capture(request: httpx.Request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=COLLECTION_RESPONSE)

    respx.post(f"{BASE}{VP}/collections").mock(side_effect=_capture)
    pc = Onecortex(url=BASE)
    pc.vector.create_collection(name="test-col", dimension=3)
    body = json.loads(captured["body"])
    assert "bm25Enabled" not in body


@respx.mock
def test_create_collection_bm25_explicit_false_is_sent():
    captured: dict[str, str] = {}

    def _capture(request: httpx.Request):
        captured["body"] = request.read().decode()
        return httpx.Response(200, json=COLLECTION_RESPONSE)

    respx.post(f"{BASE}{VP}/collections").mock(side_effect=_capture)
    pc = Onecortex(url=BASE)
    pc.vector.create_collection(name="test-col", dimension=3, bm25_enabled=False)
    body = json.loads(captured["body"])
    assert body["bm25Enabled"] is False


@respx.mock
def test_create_alias():
    route = respx.post(f"{BASE}{VP}/aliases").mock(return_value=httpx.Response(201, json=ALIAS_RESPONSE))
    pc = Onecortex(url=BASE)
    result = pc.vector.create_alias(alias="prod", collection_name="my-col-v2")
    assert result.alias == "prod"
    assert result.collection_name == "my-col-v2"
    body = json.loads(route.calls[0].request.content)
    assert body == {"alias": "prod", "collectionName": "my-col-v2"}


@respx.mock
def test_list_aliases():
    respx.get(f"{BASE}{VP}/aliases").mock(return_value=httpx.Response(200, json=ALIAS_LIST_RESPONSE))
    pc = Onecortex(url=BASE)
    result = pc.vector.list_aliases()
    assert len(result.aliases) == 2
    assert result.aliases[0].alias == "prod"
    assert result.aliases[1].collection_name == "my-col-v1"


@respx.mock
def test_describe_alias():
    respx.get(f"{BASE}{VP}/aliases/prod").mock(return_value=httpx.Response(200, json=ALIAS_RESPONSE))
    pc = Onecortex(url=BASE)
    result = pc.vector.describe_alias("prod")
    assert result.alias == "prod"
    assert result.collection_name == "my-col-v2"


@respx.mock
def test_delete_alias():
    respx.delete(f"{BASE}{VP}/aliases/prod").mock(return_value=httpx.Response(204))
    pc = Onecortex(url=BASE)
    pc.vector.delete_alias("prod")  # should not raise
