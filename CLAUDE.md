# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victorious turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match (Wins, placement, eliminations, speech), read from what is on screen and what is said. The domain vocabulary (Recording, Match, Win, Layout, HUD element, Observation, Event, Setup, Analysis, Transcript segment, Speaker) is defined in `GLOSSARY.md`; architecture decisions are in `docs/adr/`. Read both before changing behaviour. The work is tracked in epic #13.

## Architecture

Installable package in `src/victorious/` (hatchling build, console script `victorious`):

- `cli.py`: argparse entry point exposing the subcommands below.
- `__main__.py`: enables `python -m victorious`.
- `matcher.py`: internal multi-scale template matcher (scales 0.3-1.2) used as a detector for reference images such as the Victory Royale banner. Not a user-facing command.

| Subcommand | Purpose | Status |
|------------|---------|--------|
| `setup` | Produce a Layout from a Recording | stub, see epic #13 |
| `analyse` | Turn a Recording and a Layout into Events | stub, see epic #13 |
| `review` | Inspect the Events found in a Recording | stub, see epic #13 |
| `schema` | Print the JSON schema of the Analysis output | stub, see epic #13 |

Stubs print `victorious <command>: not implemented yet` to stderr and exit 1.

## Development Commands

```bash
# Install dependencies (including dev group)
uv sync --group dev

# Format and lint
uv run ruff format
uv run ruff check
uv run ruff check --fix

# Type checking (whole project, strict)
uv run mypy

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=. --cov-report=term-missing

# Run the application
uv run victorious --help
uv run victorious <subcommand>
```

Tests import the installed package (no `pythonpath` shim), so run them through `uv run`.

## Configuration

- **Ruff**: 100-character line length, Python 3.13 target
- **mypy**: strict
- **Dependencies**: Python >=3.13, OpenCV, NumPy
- **Commits and PR titles**: Conventional Commits (enforced in CI)
