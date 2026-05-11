Format the onecortex SDK codebase using ruff.

> **Note:** Lefthook pre-commit already runs `uv run ruff format --check src/ tests/` automatically. Use this command only when lefthook flags formatting issues and you want to auto-fix them before committing.

Run:
```
uv run ruff format src/ tests/
```

Report which files were reformatted. If no files changed, confirm the codebase is already properly formatted.
