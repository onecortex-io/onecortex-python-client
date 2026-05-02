import time
from typing import Any

import httpx

from .exceptions import (
    AlreadyExistsError,
    CollectionAlreadyExistsError,
    CollectionNotFoundError,
    DimensionMismatchError,
    FacetFieldInvalidError,
    FilterMalformedError,
    FilterUnsupportedOperatorError,
    GroupByFieldMissingError,
    HybridRequiresBm25Error,
    IndexNotReadyError,
    InvalidArgumentError,
    NotFoundError,
    OnecortexServerError,
    PermissionDeniedError,
    RerankerConfigError,
    RerankerRateLimitedError,
    RerankerTimeoutError,
    RerankerUpstreamError,
    SparseNotSupportedError,
    UnauthorizedError,
)

_ERROR_MAP = {
    "NOT_FOUND": NotFoundError,
    "ALREADY_EXISTS": AlreadyExistsError,
    "INVALID_ARGUMENT": InvalidArgumentError,
    "UNAUTHENTICATED": UnauthorizedError,
    "PERMISSION_DENIED": PermissionDeniedError,
    # v0.2.0 stable codes
    "DIMENSION_MISMATCH": DimensionMismatchError,
    "SPARSE_NOT_SUPPORTED": SparseNotSupportedError,
    "FILTER_MALFORMED": FilterMalformedError,
    "FILTER_UNSUPPORTED_OPERATOR": FilterUnsupportedOperatorError,
    "HYBRID_REQUIRES_BM25": HybridRequiresBm25Error,
    "GROUPBY_FIELD_MISSING": GroupByFieldMissingError,
    "FACET_FIELD_INVALID": FacetFieldInvalidError,
    "INDEX_NOT_READY": IndexNotReadyError,
    "COLLECTION_NOT_FOUND": CollectionNotFoundError,
    "COLLECTION_ALREADY_EXISTS": CollectionAlreadyExistsError,
    "RERANKER_RATE_LIMITED": RerankerRateLimitedError,
    "RERANKER_TIMEOUT": RerankerTimeoutError,
    "RERANKER_CONFIG": RerankerConfigError,
    "RERANKER_UPSTREAM": RerankerUpstreamError,
}

# HTTP statuses that should trigger a retry on a transient failure.
# 429: rate limited (incl. RERANKER_RATE_LIMITED).
# 500: generic INTERNAL.
# 504: RERANKER_TIMEOUT — retryable with caution.
# 502 (RERANKER_UPSTREAM) and 503 (RERANKER_CONFIG) are *not* retried.
_RETRYABLE_STATUSES = frozenset({429, 500, 504})


def _parse_error(response: httpx.Response) -> tuple[str, str, dict[str, Any], str | None]:
    """Return (code, message, details, request_id) parsed from an error response."""
    code = "UNKNOWN"
    message = response.text
    details: dict[str, Any] = {}
    try:
        body = response.json()
        if isinstance(body, dict):
            # Vector-style: { "error": { "code": "...", "message": "...", "details": {...} } }
            err = body.get("error")
            if isinstance(err, dict):
                code = err.get("code", "UNKNOWN")
                message = err.get("message", response.text)
                raw_details = err.get("details")
                if isinstance(raw_details, dict):
                    details = raw_details
            elif "code" in body:
                # Auth-style: { "code": "...", "msg": "..." }
                code = str(body["code"]).upper().replace("-", "_")
                message = str(body.get("msg", body.get("message", response.text)))
    except Exception:
        pass

    request_id: str | None = None
    raw_rid = details.get("requestId")
    if isinstance(raw_rid, str) and raw_rid:
        request_id = raw_rid
    else:
        header_rid = response.headers.get("x-request-id")
        if header_rid:
            request_id = header_rid

    return code, message, details, request_id


def _raise_for_response(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    code, message, details, request_id = _parse_error(response)
    exc_class = _ERROR_MAP.get(code, OnecortexServerError)
    raise exc_class(
        message,
        status_code=response.status_code,
        code=code,
        details=details,
        request_id=request_id,
    )


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
                if response.status_code in _RETRYABLE_STATUSES and attempt < len(delays):
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
