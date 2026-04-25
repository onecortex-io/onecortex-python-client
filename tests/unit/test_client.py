from onecortex import Onecortex
from onecortex.auth import AuthClient
from onecortex.vector import VectorClient

BASE = "http://test-server"


def test_onecortex_has_vector_namespace():
    client = Onecortex(url=BASE)
    assert isinstance(client.vector, VectorClient)


def test_onecortex_has_auth_namespace():
    client = Onecortex(url=BASE)
    assert isinstance(client.auth, AuthClient)


def test_onecortex_accepts_access_token():
    client = Onecortex(url=BASE, access_token="tok123")
    assert client._http._access_token == "tok123"
