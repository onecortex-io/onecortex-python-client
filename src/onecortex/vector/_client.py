from typing import Any

from .._http import HttpClient
from ._index import Collection
from .models import AliasDescription, AliasListResult, CollectionDescription


class VectorClient:
    """
    Control-plane client for vector collection management.
    Access via client.vector (e.g., client.vector.create_collection(...)).
    """

    def __init__(self, http: HttpClient, base_path: str = "/v1/vector"):
        self._http = http
        self._base_path = base_path

    def create_collection(
        self,
        name: str,
        dimension: int,
        metric: str = "cosine",
        bm25_enabled: bool = False,
        deletion_protection: str | None = None,
        tags: dict | None = None,
        **kwargs: Any,  # absorb unknown args like spec= without erroring
    ) -> CollectionDescription:
        """Create a new vector collection."""
        body: dict = {"name": name, "dimension": dimension, "metric": metric}
        if bm25_enabled:
            body["bm25_enabled"] = True
        if deletion_protection:
            body["deletion_protection"] = deletion_protection
        if tags:
            body["tags"] = tags
        response = self._http.post(f"{self._base_path}/collections", json=body)
        return CollectionDescription.model_validate(response.json())

    def describe_collection(self, name: str) -> CollectionDescription:
        response = self._http.get(f"{self._base_path}/collections/{name}")
        return CollectionDescription.model_validate(response.json())

    def list_collections(self) -> list[CollectionDescription]:
        response = self._http.get(f"{self._base_path}/collections")
        return [CollectionDescription.model_validate(i) for i in response.json().get("collections", [])]

    def delete_collection(self, name: str) -> None:
        self._http.delete(f"{self._base_path}/collections/{name}")

    def configure_collection(
        self,
        name: str,
        deletion_protection: str | None = None,
        tags: dict | None = None,
        **kwargs: Any,
    ) -> CollectionDescription:
        body: dict = {}
        if deletion_protection is not None:
            body["deletion_protection"] = deletion_protection
        if tags is not None:
            body["tags"] = tags
        response = self._http.patch(f"{self._base_path}/collections/{name}", json=body)
        return CollectionDescription.model_validate(response.json())

    def has_collection(self, name: str) -> bool:
        try:
            self.describe_collection(name)
            return True
        except Exception:
            return False

    def collection(self, name: str) -> Collection:
        """Get a handle to a specific collection for data-plane operations."""
        return Collection(http=self._http, base_path=self._base_path, name=name)

    # ── Aliases ──────────────────────────────────────────────────────────────

    def create_alias(self, alias: str, collection_name: str) -> AliasDescription:
        """Create or update an alias pointing to collection_name (upsert semantics)."""
        response = self._http.post(
            f"{self._base_path}/aliases",
            json={"alias": alias, "collectionName": collection_name},
        )
        return AliasDescription.model_validate(response.json())

    def describe_alias(self, alias: str) -> AliasDescription:
        """Fetch details for a single alias."""
        response = self._http.get(f"{self._base_path}/aliases/{alias}")
        return AliasDescription.model_validate(response.json())

    def list_aliases(self) -> AliasListResult:
        """List all aliases in the account."""
        response = self._http.get(f"{self._base_path}/aliases")
        return AliasListResult.model_validate(response.json())

    def delete_alias(self, alias: str) -> None:
        """Delete an alias. Raises NotFoundError if it does not exist."""
        self._http.delete(f"{self._base_path}/aliases/{alias}")
