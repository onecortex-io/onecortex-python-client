from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class CollectionStatus(BaseModel):
    ready: bool
    state: str


class EmbedderSpec(BaseModel):
    """Per-collection server-side embedder binding.

    When a collection is created with an EmbedderSpec, callers may send
    ``text`` instead of ``values`` on upsert and query, and the server will
    embed it using the bound backend.
    """

    model_config = ConfigDict(populate_by_name=True)

    backend: Literal["openai", "voyage", "cohere", "jina", "tei"]
    model: str
    input_type: Literal["document", "query"] | None = Field(alias="inputType", default=None)


class HybridSpec(BaseModel):
    """Custom fusion parameters for the ``hybrid`` field on /search."""

    model_config = ConfigDict(populate_by_name=True)

    alpha: float | None = None
    bm25_weight: float | None = Field(alias="bm25Weight", default=None)


class DedupSpec(BaseModel):
    """First-occurrence-wins dedup by a metadata field."""

    by: str


class CollectionDescription(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    dimension: int
    metric: str
    status: CollectionStatus
    host: str
    spec: dict = Field(default_factory=dict)
    vector_type: str = Field(alias="vectorType", default="dense")
    bm25_enabled: bool = Field(alias="bm25Enabled", default=False)
    deletion_protected: bool | None = Field(alias="deletionProtected", default=None)
    tags: dict | None = None
    embedder: EmbedderSpec | None = None


class Match(BaseModel):
    id: str
    score: float
    values: list[float] | None = None
    metadata: dict[str, Any] | None = None


class QueryResult(BaseModel):
    matches: list[Match]
    namespace: str


class UpsertResult(BaseModel):
    upserted_count: int = Field(alias="upsertedCount")
    model_config = ConfigDict(populate_by_name=True)


class FetchResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    records: list[dict]
    namespace: str
    next_cursor: str | None = Field(alias="nextCursor", default=None)


class ListResult(BaseModel):
    records: list[dict]
    namespace: str
    pagination: dict | None = None


class NamespaceSummary(BaseModel):
    record_count: int = Field(alias="recordCount")
    model_config = ConfigDict(populate_by_name=True)


class CollectionStats(BaseModel):
    namespaces: dict[str, NamespaceSummary]
    dimension: int
    collection_fullness: float = Field(alias="collectionFullness")
    total_record_count: int = Field(alias="totalRecordCount")
    model_config = ConfigDict(populate_by_name=True)


# ── Scroll / Sample ──────────────────────────────────────────────────────────


class ScrollVector(BaseModel):
    id: str
    values: list[float] | None = None
    metadata: dict[str, Any] | None = None


class ScrollResult(BaseModel):
    """Returned by scroll() and sample(). next_cursor is None on the last page."""

    records: list[ScrollVector]
    namespace: str
    next_cursor: str | None = Field(None, alias="nextCursor")
    model_config = ConfigDict(populate_by_name=True)


# ── Batch Query ──────────────────────────────────────────────────────────────


class BatchQueryResult(BaseModel):
    """One QueryResult per input query, in the same order."""

    results: list[QueryResult]


# ── GroupBy ──────────────────────────────────────────────────────────────────


class GroupedMatch(BaseModel):
    key: str
    matches: list[Match]


class GroupedQueryResult(BaseModel):
    """Returned by query() when group_by is provided."""

    groups: list[GroupedMatch]
    namespace: str
    grouped: bool = True


# ── Recommendations ──────────────────────────────────────────────────────────


class RecommendResult(BaseModel):
    matches: list[Match]
    namespace: str


# ── Faceted Counts ───────────────────────────────────────────────────────────


class FacetEntry(BaseModel):
    value: str
    count: int


class FacetResult(BaseModel):
    """Returned by facet_counts(). Each entry is a distinct metadata value with its record count."""

    facets: list[FacetEntry]
    field: str
    namespace: str


# ── Aliases ──────────────────────────────────────────────────────────────────


class AliasDescription(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    alias: str
    collection_name: str = Field(alias="collectionName")


class AliasListResult(BaseModel):
    aliases: list[AliasDescription]
