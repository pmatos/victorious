# Victorious

Turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match: Wins, final placement, eliminations and speech.

The four subcommands below are placeholders. Each prints a not-implemented message and exits 1 until its ticket in [epic #13](https://github.com/pmatos/victorious/issues/13) lands.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Usage

```bash
uv run victorious --help
```

| Subcommand | Purpose | Status |
|------------|---------|--------|
| `setup` | Produce a Layout from a Recording | placeholder |
| `analyse` | Turn a Recording and a Layout into Events | placeholder |
| `review` | Inspect the Events found in a Recording | placeholder |
| `schema` | Print the JSON schema of the Analysis output | placeholder |

See [GLOSSARY.md](GLOSSARY.md) for the terms used above and [docs/adr](docs/adr) for design decisions.

## Development

```bash
uv sync --group dev
uv run ruff format
uv run ruff check
uv run mypy
uv run pytest
```

## Requirements

- Python >= 3.13
- OpenCV (opencv-python)
- NumPy
