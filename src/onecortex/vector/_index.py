from .._http import HttpClient
from .models import (
    BatchQueryResult,
    CollectionStats,
    FacetResult,
    FetchResult,
    GroupedMatch,
    GroupedQueryResult,
    ListResult,
    QueryResult,
    RecommendResult,
    ScrollResult,
    UpsertResult,
)


class Collection:
    """Handles all data-plane operations for a specific collection."""

    def __init__(self, http: HttpClient, base_path: str, name: str):
        self._http = http
        self._name = name
        self._base = f"{base_path}/collections/{name}"

    def upsert(
        self,
        vectors: list[dict],
        namespace: str = "",
    ) -> UpsertResult:
        """
        Upsert records into the collection.
        Each vector dict: {"id": str, "values": list[float], "metadata": dict (optional), "text": str (optional)}
        Note: "sparseValues" key is accepted and silently ignored by the server.
        """
        response = self._http.post(
            f"{self._base}/records/upsert",
            json={"records": vectors, "namespace": namespace},
        )
        return UpsertResult.model_validate(response.json())

    def upsert_batch(
        self,
        vectors: list[dict],
        namespace: str = "",
        batch_size: int = 200,
    ) -> int:
        """
        Upsert a large list of records in batches.
        Returns total upserted count.
        """
        total = 0
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            result = self.upsert(batch, namespace=namespace)
            total += result.upserted_count
        return total

    def fetch(
        self,
        ids: list[str],
        namespace: str = "",
    ) -> FetchResult:
        """Fetch records by ID."""
        response = self._http.post(
            f"{self._base}/records/fetch",
            json={"ids": ids, "namespace": namespace},
        )
        return FetchResult.model_validate(response.json())

    def fetch_by_metadata(
        self,
        filter: dict,
        namespace: str = "",
        limit: int = 100,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> FetchResult:
        """Fetch records matching a metadata filter (Onecortex extension)."""
        response = self._http.post(
            f"{self._base}/records/fetch_by_metadata",
            json={
                "filter": filter,
                "namespace": namespace,
                "limit": limit,
                "includeValues": include_values,
                "includeMetadata": include_metadata,
            },
        )
        return FetchResult.model_validate(response.json())

    def delete(
        self,
        ids: list[str] | None = None,
        filter: dict | None = None,
        delete_all: bool = False,
        namespace: str = "",
    ) -> None:
        """Delete records by IDs, by metadata filter, or all in namespace."""
        body: dict = {"namespace": namespace}
        if delete_all:
            body["deleteAll"] = True
        elif ids is not None:
            body["ids"] = ids
        elif filter is not None:
            body["filter"] = filter
        else:
            raise ValueError("Provide ids, filter, or delete_all=True")
        self._http.post(f"{self._base}/records/delete", json=body)

    def update(
        self,
        id: str,
        values: list[float] | None = None,
        set_metadata: dict | None = None,
        text: str | None = None,
        namespace: str = "",
    ) -> None:
        """Update values and/or metadata for a single record. Metadata is merged, not replaced."""
        body: dict = {"id": id, "namespace": namespace}
        if values is not None:
            body["values"] = values
        if set_metadata is not None:
            body["setMetadata"] = set_metadata
        if text is not None:
            body["text"] = text
        self._http.post(f"{self._base}/records/update", json=body)

    def query(
        self,
        vector: list[float],
        top_k: int = 10,
        namespace: str = "",
        filter: dict | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
        id: str | None = None,
        rerank: dict | None = None,
        score_threshold: float | None = None,
        group_by: dict | None = None,
    ) -> QueryResult | GroupedQueryResult:
        """
        Search for similar records using dense ANN.

        Args:
            vector: Query vector (must match collection dimension).
            top_k: Number of results (max 10 000).
            filter: Metadata filter using the same DSL as fetch_by_metadata().
            id: Query by record ID instead of a raw vector.
            rerank: Optional reranking options dict with keys:
                query (str, required): Natural-language reranker query.
                topN (int, optional): Results after reranking. Defaults to top_k.
                rankField (str, optional): Metadata field to rank against. Default "text".
                model (str, optional): Per-request model override.
            score_threshold: Minimum similarity score; results below are dropped.
            group_by: Group results by a metadata field. Dict keys:
                field (str, required): Metadata field to group on.
                limit (int, optional): Max number of groups returned.
                groupSize (int, optional): Max matches per group.
                When set, returns GroupedQueryResult instead of QueryResult.
        """
        body: dict = {
            "topK": top_k,
            "namespace": namespace,
            "includeValues": include_values,
            "includeMetadata": include_metadata,
        }
        if id is not None:
            body["id"] = id
        else:
            body["vector"] = vector
        if filter is not None:
            body["filter"] = filter
        if rerank is not None:
            body["rerank"] = rerank
        if score_threshold is not None:
            body["scoreThreshold"] = score_threshold
        if group_by is not None:
            body["groupBy"] = group_by
            body["includeMetadata"] = True  # server requires metadata to resolve group field

        response = self._http.post(f"{self._base}/query", json=body)
        data = response.json()

        if group_by is not None:
            raw_groups = data.get("groups", [])
            return GroupedQueryResult(
                groups=[GroupedMatch(**g) for g in raw_groups],
                namespace=data.get("namespace", ""),
            )
        return QueryResult.model_validate(data)

    def query_hybrid(
        self,
        vector: list[float],
        text: str,
        top_k: int = 10,
        alpha: float = 0.5,
        namespace: str = "",
        filter: dict | None = None,
        include_metadata: bool = False,
        include_values: bool = False,
        rerank: dict | None = None,
        score_threshold: float | None = None,
    ) -> QueryResult:
        """
        Hybrid ANN + BM25 query with Reciprocal Rank Fusion.

        Args:
            vector: Dense query vector (must match collection dimension).
            text:   BM25 query text.
            top_k:  Number of results to return (max 10000).
            alpha:  Dense weight [0.0, 1.0]. 0.5 = equal blend.
            filter: Metadata filter (same DSL as query()).
            namespace: Namespace to search within.
            include_metadata: Include metadata in results.
            include_values:   Include vector values in results.
            rerank: Optional reranking options dict.
            score_threshold: Minimum similarity score; results below are dropped.
        """
        body: dict = {
            "vector": vector,
            "text": text,
            "topK": top_k,
            "alpha": alpha,
            "namespace": namespace,
            "includeMetadata": include_metadata,
            "includeValues": include_values,
        }
        if filter is not None:
            body["filter"] = filter
        if rerank is not None:
            body["rerank"] = rerank
        if score_threshold is not None:
            body["scoreThreshold"] = score_threshold
        response = self._http.post(f"{self._base}/query/hybrid", json=body)
        return QueryResult.model_validate(response.json())

    def scroll(
        self,
        namespace: str = "",
        filter: dict | None = None,
        limit: int = 100,
        cursor: str | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> ScrollResult:
        """
        Paginate over all records in the collection using cursor-based iteration.

        Pass scroll_result.next_cursor into cursor= to advance to the next page.
        Returns ScrollResult with next_cursor=None on the last page.
        """
        body: dict = {
            "namespace": namespace,
            "limit": limit,
            "includeValues": include_values,
            "includeMetadata": include_metadata,
        }
        if filter is not None:
            body["filter"] = filter
        if cursor is not None:
            body["cursor"] = cursor
        response = self._http.post(f"{self._base}/records/scroll", json=body)
        return ScrollResult.model_validate(response.json())

    def sample(
        self,
        size: int = 10,
        namespace: str = "",
        filter: dict | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
    ) -> ScrollResult:
        """
        Return a random sample of up to `size` records from the collection.
        Supports the same filter DSL as query().
        """
        body: dict = {
            "namespace": namespace,
            "size": size,
            "includeValues": include_values,
            "includeMetadata": include_metadata,
        }
        if filter is not None:
            body["filter"] = filter
        response = self._http.post(f"{self._base}/sample", json=body)
        return ScrollResult.model_validate(response.json())

    def query_batch(self, queries: list[dict]) -> BatchQueryResult:
        """
        Execute up to 10 queries concurrently in a single round-trip.

        Each dict in queries accepts the same camelCase keys the server expects:
        vector, topK, filter, includeValues, includeMetadata, scoreThreshold, groupBy, rerank.
        Results are returned in the same order as the input queries.
        """
        if not queries:
            raise ValueError("queries must not be empty")
        if len(queries) > 10:
            raise ValueError("queries cannot exceed 10 entries")
        response = self._http.post(f"{self._base}/query/batch", json={"queries": queries})
        return BatchQueryResult.model_validate(response.json())

    def recommend(
        self,
        positive_ids: list[str],
        negative_ids: list[str] | None = None,
        top_k: int = 10,
        namespace: str = "",
        filter: dict | None = None,
        include_values: bool = False,
        include_metadata: bool = False,
        score_threshold: float | None = None,
    ) -> RecommendResult:
        """
        Find similar records from positive example IDs, optionally steered away
        from negative example IDs. Input IDs are always excluded from results.

        Algorithm: mean(positives) - mean(negatives) used as the synthetic query vector.
        """
        body: dict = {
            "positiveIds": positive_ids,
            "negativeIds": negative_ids or [],
            "topK": top_k,
            "namespace": namespace,
            "includeValues": include_values,
            "includeMetadata": include_metadata,
        }
        if filter is not None:
            body["filter"] = filter
        if score_threshold is not None:
            body["scoreThreshold"] = score_threshold
        response = self._http.post(f"{self._base}/recommend", json=body)
        return RecommendResult.model_validate(response.json())

    def list(
        self,
        namespace: str = "",
        prefix: str = "",
        limit: int = 100,
        pagination_token: str | None = None,
    ) -> ListResult:
        """List record IDs in a namespace, optionally filtered by prefix."""
        params: dict = {"namespace": namespace, "limit": str(limit)}
        if prefix:
            params["prefix"] = prefix
        if pagination_token:
            params["paginationToken"] = pagination_token
        response = self._http.get(f"{self._base}/records/list", params=params)
        return ListResult.model_validate(response.json())

    def facet_counts(
        self,
        field: str,
        filter: dict | None = None,
        namespace: str = "",
        limit: int = 20,
    ) -> FacetResult:
        """
        Return aggregated counts of distinct metadata values for a field.

        Results are ordered by count descending. Records missing the requested
        field are excluded. Supports the same filter DSL as query().

        Args:
            field: Metadata field to facet on (letters, digits, underscores, dots).
            filter: Optional metadata filter to restrict which records are counted.
            namespace: Namespace to scope the operation to.
            limit: Maximum number of facet entries returned (1-100, default 20).
        """
        body: dict = {"field": field, "namespace": namespace, "limit": limit}
        if filter is not None:
            body["filter"] = filter
        response = self._http.post(f"{self._base}/facets", json=body)
        return FacetResult.model_validate(response.json())

    def describe_collection_stats(self) -> CollectionStats:
        """Get record counts per namespace."""
        response = self._http.post(f"{self._base}/describe_collection_stats", json={})
        return CollectionStats.model_validate(response.json())
