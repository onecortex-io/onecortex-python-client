# OneCortex Python SDK

Python client for the OneCortex platform. Built with httpx + pydantic, packaged with hatchling, managed with uv.

## Commands

```bash
uv sync                                    # Install dependencies
uv run pytest tests/unit/ -v               # Unit tests (default, no server needed)
uv run pytest tests/integration/ -v        # Integration tests (requires live server)
uv run ruff check src/ tests/              # Lint
uv run ruff check src/ tests/ --fix        # Lint with auto-fix
uv run ruff format src/ tests/             # Format
uv run mypy src/onecortex/                 # Type check
uv build                                   # Build package
```

## Architecture

```
Onecortex(url, api_key)          # Facade — src/onecortex/_client.py
├── .vector: VectorClient        # Control plane — src/onecortex/vector/_client.py
│   └── .index(name) -> Index    # Data plane — src/onecortex/vector/_index.py
├── .auth: AuthClient            # Stub — src/onecortex/auth/_client.py
└── _http: HttpClient            # Shared HTTP — src/onecortex/_http.py
```

All HTTP goes through `HttpClient.request()` — never use httpx directly in service clients. HttpClient handles auth headers and retry (1s/2s/4s backoff on 429/5xx).

## Conventions

- **Package manager**: Always use `uv run` — never bare `python` or `pip`
- **Commits**: Conventional commits — `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`, `build:`, `ci:`. Use `!` for breaking changes
- **Pydantic models**: Use `Field(alias="camelCase")` + `ConfigDict(populate_by_name=True)` for JSON mapping
- **Exports**: All public API must be re-exported in `src/onecortex/__init__.py` and listed in `__all__`
- **Tests**: Unit tests use `respx` to mock HTTP. Integration tests require `ONECORTEX_HOST` and `ONECORTEX_API_KEY` env vars
- **Exceptions**: All inherit from `OnecortexError`. Error codes from server map to specific exception classes in `exceptions.py`
- **Version**: Single source of truth is `version` in `pyproject.toml`

---

## Cross-Service Context

This SDK implements the Python interface to `onecortex-vector` (fully implemented)
and `onecortex-auth` (stub at `src/onecortex/auth/_client.py`).

SDK parity rule: every public method in this client must have an equivalent in
`onecortex-typescript-client` with the same logical name (mapped snake_case ↔ camelCase),
same parameters, and identical error behaviour. Never add a method to one SDK
without adding it to the other.

When the backend API changes, use `/sync-clients <description>` from the org root
to update both SDKs together.

See `../CLAUDE.md` for the full parity rules, dependency graph, and slash commands.
