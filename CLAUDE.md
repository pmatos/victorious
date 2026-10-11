# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victorious turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match (Wins, placement, eliminations, speech), read from what is on screen and what is said. The domain vocabulary (Recording, Match, Win, Layout, HUD element, Observation, Event, Setup, Analysis, Transcript segment, Speaker) is defined in `GLOSSARY.md`; architecture decisions are in `docs/adr/`. Read both before changing behaviour. The work is tracked in epic #13.

## Architecture

Installable package in `src/victorious/` (hatchling build, console script `victorious`):

- `cli.py`: argparse entry point exposing the subcommands below.
- `__main__.py`: enables `python -m victorious`.
- `matcher.py`: internal multi-scale template matcher (scales 0.3-1.2) used as a detector for reference images such as the Victory Royale banner. Not a user-facing command.
- `layout.py`: pydantic `Layout` (YAML, normalized rects) and `load_layout`.
- `schema.py`: pydantic output models (`AnalysisOutput`, `Match`, `Event`), `SCHEMA_VERSION`, `json_schema()`.
- `video.py`: `probe` (ffprobe), `sample_chunk` (one BGR frame per second of a whole-second chunk through an ffmpeg raw pipe) and `sample_window` (a short seek-based decode at a higher rate, used to bisect counter jumps).
- `references.py` and `data/`: reference images shipped in the package (HUD icons, Victory Royale banner, `#` glyph).
- `screens.py`: pure per-frame detection, `hud_score`, `results_reading` and `counter_masks`; thresholds `HUD_THRESHOLD`/`RESULTS_THRESHOLD`.
- `ocr.py`: per-glyph Tesseract reading of a binary mask (`tesseract` CLI via subprocess).
- `counters.py`: `ReuseTracker` (the skip-OCR rule: read again only when the pixels differ from the last OCR'd ones), `CounterReader` (the serial reader built on it) and `CounterTracker` (turns readings into change Events and `unreadable` spans, and records jumps of more than one as `Gap`s for a resolver).
- `resample.py`: `resolve_jump` (pure: times each step of a jump from 4 fps samples) and `JumpResolver` (decodes each Gap's window, pooled per Match close). `analyse` seeks only for ambiguous gaps.
- `observations.py`: `SecondObservation` (everything read at one second), per-chunk extraction with packed counter masks, JSON (de)serialisation and the `--observations` records.
- `parallel.py`: the chunk plan and `spawn` process pool for frame facts (abort flag, error transport), and the OCR thread pool for counters.
- `cache.py`: Observation cache keyed by Recording fingerprint, Layout hash and `OBSERVATION_VERSION` (bump it when readings or thresholds change), plus the bisect windows.
- `analysis.py`: `analyse(path, layout, jobs=, chunk_seconds=, cache=)` and `run_machine`, the state machine that turns per-second Observations into Matches and Events. A Match opens on the HUD and closes only on a results screen. Output must not depend on `jobs` or chunk size: the skip-OCR rule is replayed with `ReuseTracker` over the exact schedule of counter reads the machine produces, instead of running per chunk.

| Subcommand | Purpose | Status |
|------------|---------|--------|
| `setup` | Produce a Layout from a Recording | stub, see epic #13 |
| `analyse` | Turn a Recording and a Layout into Matches with Events | working: Matches, Win, placement, HUD counters (speech: later tickets) |
| `review` | Inspect the Events found in a Recording | stub, see epic #13 |
| `schema` | Print the JSON schema of the Analysis output | working |

`setup` and `review` are stubs: they print `victorious <command>: not implemented yet` to stderr and exit 1. `analyse` takes `--jobs` (default: CPU count), `--cache-dir`/`--no-cache`, `--observations out.jsonl` and a hidden `--chunk-seconds`. Errors in `analyse` print `victorious analyse: <reason>` to stderr, exit 1 and write nothing.

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

## Test fixtures

Tests build their Recordings on demand with `tests.synthetic`; no video is committed.

```python
from tests.synthetic import RecordingSpec, build_recording, match_shots

spec = RecordingSpec(duration=35.0, shots=match_shots(5.0, [(0.0, 0, 0, 98)], 25.0))
build_recording(spec, tmp_path / "recording.mkv")
```

- `ffmpeg`, `ffprobe` and `tesseract` must be on `PATH` (tests fail with a clear message otherwise; CI installs them).
- Scenes are composited from small real-HUD crops: icons, banner and `#` live in `src/victorious/data/` (shared with the product), digits in `tests/fixtures/crops/`; `tests/fixtures/make_crops.py` records where each came from and re-cuts them from local Recordings (never run in CI).
- `ELEMENT_RECTS` gives the normalized rectangle of each HUD element, for tests that inspect decoded frames; `write_layout(path)` writes the matching hand-written Layout YAML for `analyse` tests.

## Configuration

- **Ruff**: 100-character line length, Python 3.13 target
- **mypy**: strict
- **Dependencies**: Python >=3.13, OpenCV, NumPy, pydantic, PyYAML; runtime binaries ffmpeg, ffprobe, tesseract
- **Commits and PR titles**: Conventional Commits (enforced in CI)
