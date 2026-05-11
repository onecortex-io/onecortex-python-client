Run the test suite for the onecortex SDK.

> **Note:** Lefthook pre-commit already runs `uv run pytest tests/unit/ -v` automatically, and pre-push runs `uv run pytest tests/integration/ -v`. Use this command when you want to run a **specific** test, get detailed failure diagnosis, or explicitly request integration tests.

## Default: Unit Tests

Run unit tests (no server required):

```
uv run pytest tests/unit/ -v
```

## Integration Tests

Only run integration tests if the user explicitly asks for them (e.g., "run all tests", "run integration tests").

Before running integration tests, check if `ONECORTEX_HOST` and `ONECORTEX_API_KEY` environment variables are set by running `echo $ONECORTEX_HOST`. If not set, warn the user that integration tests require a live server and these env vars, and skip them.

If env vars are set:
```
uv run pytest tests/integration/ -v
```

## On Failure

When tests fail:
1. Read the failing test file to understand what is being tested
2. Read the source code under test to understand the expected behavior
3. Diagnose the root cause of the failure
4. Suggest a specific fix with code changes

Unit tests use `respx` for HTTP mocking — check that mock routes and response payloads match the current API contract.
