Run linting and essential tests in sequence:

1. Run `uv run ruff check vla_foundry/` and report any issues
2. Run `uv run pytest tests/essential/ -x -v` and report results
3. Summarize: number of lint errors, tests passed/failed, and any action items