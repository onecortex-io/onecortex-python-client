import time
from typing import Any

import httpx

from .exceptions import (
    AlreadyExistsError,
    InvalidArgumentError,
    NotFoundError,
    OnecortexServerError,
    PermissionDeniedError,
    UnauthorizedError,
)

_ERROR_MAP = {
    "NOT_FOUND": NotFoundError,
    "ALREADY_EXISTS": AlreadyExistsError,
    "INVALID_ARGUMENT": InvalidArgumentError,
    "UNAUTHENTICATED": UnauthorizedError,
    "PERMISSION_DENIED": PermissionDeniedError,
}


def _raise_for_response(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    try:
        body = response.json()
        # Vector-style: { "error": { "code": "...", "message": "..." } }
        if "error" in body and isinstance(body["error"], dict):
            code = body["error"].get("code", "UNKNOWN")
            message = body["error"].get("message", response.text)
        # Auth-style: { "code": "...", "msg": "..." }
        elif "code" in body:
            code = str(body["code"]).upper().replace("-", "_")
            message = body.get("msg", body.get("message", response.text))
        else:
            code = "UNKNOWN"
            message = response.text
    except Exception:
        code = "UNKNOWN"
        message = response.text

    exc_class = _ERROR_MAP.get(code, OnecortexServerError)
    raise exc_class(message, status_code=response.status_code)


class HttpClient:
    """Synchronous httpx client with JWT Bearer auth and token auto-refresh."""

    def __init__(self, url: str, access_token: str | None = None):
        self._base_url = url.rstrip("/")
        self._client = httpx.Client(
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        self._access_token: str | None = None
        self._refresh_token: str | None = None
        self._expires_at: int | None = None
        self._refreshing = False
        if access_token:
            self.set_token(access_token)

    def set_token(
        self,
        access_token: str,
        refresh_token: str | None = None,
        expires_at: int | None = None,
    ) -> None:
        self._access_token = access_token
        self._refresh_token = refresh_token
        self._expires_at = expires_at

    def clear_token(self) -> None:
        self._access_token = None
        self._refresh_token = None
        self._expires_at = None

    def _maybe_refresh(self) -> None:
        if self._refreshing or self._refresh_token is None or self._expires_at is None:
            return
        if self._expires_at - int(time.time()) >= 60:
            return
        self._refreshing = True
        try:
            resp = self._client.request(
                "POST",
                f"{self._base_url}/auth/token",
                json={"grant_type": "refresh_token", "refresh_token": self._refresh_token},
            )
            if resp.status_code == 200:
                data = resp.json()
                self.set_token(
                    data["access_token"],
                    data.get("refresh_token"),
                    data.get("expires_at"),
                )
        except Exception:
            pass
        finally:
            self._refreshing = False

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        self._maybe_refresh()
        url = f"{self._base_url}{path}"
        if self._access_token:
            extra = {"Authorization": f"Bearer {self._access_token}"}
            existing = kwargs.pop("headers", {})
            kwargs["headers"] = {**extra, **existing}

        delays = [1, 2, 4]
        last_exc: httpx.Response | Exception | None = None
        for attempt, delay in enumerate([0, *delays]):
            if delay:
                time.sleep(delay)
            try:
                response = self._client.request(method, url, **kwargs)
                if (response.status_code in (429,) or response.status_code >= 500) and attempt < len(delays):
                    last_exc = response
                    continue
                _raise_for_response(response)
                return response
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_exc = e
                continue
        if isinstance(last_exc, httpx.Response):
            _raise_for_response(last_exc)
        raise OnecortexServerError(f"Request failed after retries: {last_exc}")

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("GET", path, **kwargs)

    def post(self, path: str, json: dict | None = None, **kwargs: Any) -> httpx.Response:
        return self.request("POST", path, json=json, **kwargs)

    def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", path, **kwargs)

    def patch(self, path: str, json: dict | None = None, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", path, json=json, **kwargs)

    def put(self, path: str, json: dict | None = None, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", path, json=json, **kwargs)
