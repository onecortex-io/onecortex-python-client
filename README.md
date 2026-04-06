# onecortex

Python client for the [OneCortex](https://github.com/onecortex-io) platform.

## Installation

```bash
pip install onecortex
# or with uv
uv add onecortex
```

Requires Python 3.11+.

## Quick Start

```python
from onecortex import Onecortex

client = Onecortex(url="https://your-project.onecortex.io", api_key="your-api-key")

# Vector database
client.vector.create_collection(name="my-collection", dimension=1536, metric="cosine")
col = client.vector.collection("my-collection")

col.upsert(vectors=[
    {"id": "vec-1", "values": [0.1, 0.2, 0.3, ...], "metadata": {"genre": "sci-fi"}},
    {"id": "vec-2", "values": [0.4, 0.5, 0.6, ...], "metadata": {"genre": "fantasy"}},
])

results = col.query(vector=[0.1, 0.2, 0.3, ...], top_k=5, include_metadata=True)
for match in results.matches:
    print(f"{match.id}: score={match.score}")

# Auth (coming soon)
# client.auth.sign_in(email, password)
```

## Available Services

| Service | Namespace | Status |
|---------|-----------|--------|
| Vector Database | `client.vector` | Available |
| Auth | `client.auth` | Coming soon |
| Storage | `client.storage` | Planned |
| Database REST | `client.db` | Planned |
| Realtime | `client.realtime` | Planned |

## Vector Database

For comprehensive vector database documentation including hybrid search, reranking, metadata filtering, and more, see [docs/vector-api.md](docs/vector-api.md).

### Key Features

- Dense ANN similarity search (cosine, euclidean, dotproduct)
- Hybrid search (dense + BM25 keyword matching)
- Reranking with natural language queries
- Metadata filtering with rich query DSL
- Namespace-based data isolation
- Automatic retry with exponential backoff

## Error Handling

```python
from onecortex import (
    OnecortexError,
    NotFoundError,
    AlreadyExistsError,
    InvalidArgumentError,
    UnauthorizedError,
    PermissionDeniedError,
)

try:
    client.vector.describe_collection("nonexistent")
except NotFoundError as e:
    print(f"Collection not found: {e} (status={e.status_code})")
except OnecortexError as e:
    print(f"Error: {e}")
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
