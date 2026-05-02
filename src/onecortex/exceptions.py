from typing import Any


class OnecortexError(Exception):
    """Base exception for all Onecortex SDK errors."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        *,
        code: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
    ):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.details: dict[str, Any] = details or {}
        self.request_id = request_id


class NotFoundError(OnecortexError):
    """Raised when a resource (index, vector) does not exist."""


class AlreadyExistsError(OnecortexError):
    """Raised when trying to create a resource that already exists."""


class InvalidArgumentError(OnecortexError):
    """Raised for invalid request parameters."""


class UnauthorizedError(OnecortexError):
    """Raised when the API key is missing or invalid."""


class PermissionDeniedError(OnecortexError):
    """Raised when the key lacks access to the requested namespace."""


class OnecortexServerError(OnecortexError):
    """Raised for unexpected server errors (5xx)."""


# --- v0.2.0 typed error subclasses (onecortex-vector stable codes) ---


class DimensionMismatchError(InvalidArgumentError):
    """Vector length does not match the collection dimension."""


class SparseNotSupportedError(InvalidArgumentError):
    """Upsert payload contained `sparseValues`, which is not supported."""


class FilterMalformedError(InvalidArgumentError):
    """Metadata filter could not be parsed."""


class FilterUnsupportedOperatorError(InvalidArgumentError):
    """Metadata filter used an operator the server does not support."""


class HybridRequiresBm25Error(InvalidArgumentError):
    """Hybrid query issued against a collection without BM25 indexing."""


class GroupByFieldMissingError(InvalidArgumentError):
    """`groupBy` field is absent on every matched record."""


class FacetFieldInvalidError(InvalidArgumentError):
    """Facet field is not faceted or otherwise invalid."""


class CollectionNotFoundError(NotFoundError):
    """Named collection does not exist."""


class CollectionAlreadyExistsError(AlreadyExistsError):
    """Collection name is already taken."""


class IndexNotReadyError(OnecortexError):
    """Collection exists but its index is still building."""


class RerankerRateLimitedError(OnecortexServerError):
    """Reranker upstream returned 429 — retryable with backoff."""


class RerankerUpstreamError(OnecortexServerError):
    """Reranker upstream returned a non-2xx, failed to connect, or response was unparseable."""


class RerankerConfigError(OnecortexServerError):
    """Reranker is misconfigured (e.g., missing API key)."""


class RerankerTimeoutError(OnecortexServerError):
    """Reranker upstream timed out — retryable with caution."""
