from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CollectionStatus(BaseModel):
    ready: bool
    state: str


class CollectionDescription(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str
    dimension: int
    metric: str
    status: CollectionStatus
    host: str
    spec: dict = Field(default_factory=dict)
    vector_type: str = "dense"
    tags: dict | None = None


class Match(BaseModel):
    id: str
    score: float
    values: list[float] | None = None
    metadata: dict[str, Any] | None = None


class QueryResult(BaseModel):
    matches: list[Match]
    namespace: str
    results: list = Field(default_factory=list)  # deprecated legacy field


class UpsertResult(BaseModel):
    upserted_count: int = Field(alias="upsertedCount")
    model_config = ConfigDict(populate_by_name=True)


class FetchResult(BaseModel):
    records: dict[str, Any]
    namespace: str


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
    group: str
    matches: list[Match]


class GroupedQueryResult(BaseModel):
    """Returned by query() when group_by is provided."""

    groups: list[GroupedMatch]
    namespace: str


# ── Recommendations ──────────────────────────────────────────────────────────


class RecommendResult(BaseModel):
    matches: list[Match]
    namespace: str


# ── Aliases ──────────────────────────────────────────────────────────────────


class AliasDescription(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    alias: str
    collection_name: str = Field(alias="collectionName")


class AliasListResult(BaseModel):
    aliases: list[AliasDescription]
