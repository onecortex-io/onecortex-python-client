import httpx
import pytest
import respx

from onecortex import Onecortex
from onecortex.exceptions import NotFoundError

BASE = "http://test-server:8080"
VP = "/v1/vector"

COLLECTION_RESPONSE = {
    "name": "test-col",
    "dimension": 3,
    "metric": "cosine",
    "status": {"ready": True, "state": "Ready"},
    "host": "test-server:8080",
}


@respx.mock
def test_create_collection():
    respx.post(f"{BASE}{VP}/collections").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE, api_key="key123")
    col = pc.vector.create_collection(name="test-col", dimension=3, metric="cosine")
    assert col.name == "test-col"
    assert col.dimension == 3


@respx.mock
def test_create_collection_ignores_spec():
    respx.post(f"{BASE}{VP}/collections").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE, api_key="key123")
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
    pc = Onecortex(url=BASE, api_key="key123")
    col = pc.vector.describe_collection("test-col")
    assert col.metric == "cosine"


@respx.mock
def test_list_collections():
    respx.get(f"{BASE}{VP}/collections").mock(
        return_value=httpx.Response(200, json={"collections": [COLLECTION_RESPONSE]})
    )
    pc = Onecortex(url=BASE, api_key="key123")
    collections = pc.vector.list_collections()
    assert len(collections) == 1
    assert collections[0].name == "test-col"


@respx.mock
def test_delete_collection():
    respx.delete(f"{BASE}{VP}/collections/test-col").mock(return_value=httpx.Response(202))
    pc = Onecortex(url=BASE, api_key="key123")
    pc.vector.delete_collection("test-col")  # should not raise


@respx.mock
def test_has_collection_true():
    respx.get(f"{BASE}{VP}/collections/test-col").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE, api_key="key123")
    assert pc.vector.has_collection("test-col") is True


@respx.mock
def test_has_collection_false():
    respx.get(f"{BASE}{VP}/collections/missing").mock(
        return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "not found"}})
    )
    pc = Onecortex(url=BASE, api_key="key123")
    assert pc.vector.has_collection("missing") is False


@respx.mock
def test_configure_collection():
    respx.patch(f"{BASE}{VP}/collections/test-col").mock(return_value=httpx.Response(200, json=COLLECTION_RESPONSE))
    pc = Onecortex(url=BASE, api_key="key123")
    result = pc.vector.configure_collection("test-col", tags={"env": "prod"})
    assert result.name == "test-col"


@respx.mock
def test_not_found_raises():
    respx.get(f"{BASE}{VP}/collections/missing").mock(
        return_value=httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "not found"}})
    )
    pc = Onecortex(url=BASE, api_key="key123")
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
def test_create_alias():
    import json

    route = respx.post(f"{BASE}{VP}/aliases").mock(return_value=httpx.Response(201, json=ALIAS_RESPONSE))
    pc = Onecortex(url=BASE, api_key="key123")
    result = pc.vector.create_alias(alias="prod", collection_name="my-col-v2")
    assert result.alias == "prod"
    assert result.collection_name == "my-col-v2"
    body = json.loads(route.calls[0].request.content)
    assert body == {"alias": "prod", "collectionName": "my-col-v2"}


@respx.mock
def test_list_aliases():
    respx.get(f"{BASE}{VP}/aliases").mock(return_value=httpx.Response(200, json=ALIAS_LIST_RESPONSE))
    pc = Onecortex(url=BASE, api_key="key123")
    result = pc.vector.list_aliases()
    assert len(result.aliases) == 2
    assert result.aliases[0].alias == "prod"
    assert result.aliases[1].collection_name == "my-col-v1"


@respx.mock
def test_describe_alias():
    respx.get(f"{BASE}{VP}/aliases/prod").mock(return_value=httpx.Response(200, json=ALIAS_RESPONSE))
    pc = Onecortex(url=BASE, api_key="key123")
    result = pc.vector.describe_alias("prod")
    assert result.alias == "prod"
    assert result.collection_name == "my-col-v2"


@respx.mock
def test_delete_alias():
    respx.delete(f"{BASE}{VP}/aliases/prod").mock(return_value=httpx.Response(204))
    pc = Onecortex(url=BASE, api_key="key123")
    pc.vector.delete_alias("prod")  # should not raise
