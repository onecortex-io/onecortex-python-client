from ._http import HttpClient
from .auth import AuthClient
from .vector import VectorClient


class Onecortex:
    """Unified OneCortex client. Access services via namespaces: .vector, .auth"""

    def __init__(self, url: str, access_token: str | None = None) -> None:
        self._http = HttpClient(url=url, access_token=access_token)
        self.vector = VectorClient(http=self._http, base_path="/vector/v1")
        self.auth = AuthClient(http=self._http, base_path="/auth")
