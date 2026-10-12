# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victorious turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match (Wins, placement, eliminations, speech), read from what is on screen and what is said. The domain vocabulary (Recording, Match, Win, Layout, HUD element, Observation, Event, Setup, Analysis, Transcript segment, Speaker) is defined in `GLOSSARY.md`; architecture decisions are in `docs/adr/`. Read both before changing behaviour. The work is tracked in epic #13.

## Architecture

Installable package in `src/victorious/` (hatchling build, console script `victorious`):

- `cli.py`: argparse entry point exposing the subcommands below.
- `__main__.py`: enables `python -m victorious`.
- `matcher.py`: internal multi-scale template matcher (scales 0.3-1.2) used as a detector for reference images such as the Victory Royale banner. Not a user-facing command.
- `layout.py`: pydantic `Layout` (YAML, normalized rects), `Layout.fits` (aspect ratio), `load_layout` and `load_layouts` (a file or a directory of Layouts, as `LoadedLayout`s).
- `schema.py`: pydantic output models (`AnalysisOutput`, `Match`, `Event`), `SCHEMA_VERSION`, `json_schema()`.
- `video.py`: `probe` (ffprobe, with the container `creation_time`), `require_complete` (one tail seek that fails a truncated Recording), `sample_chunk` (one BGR frame per second of a whole-second chunk through an ffmpeg raw pipe) and `sample_window` (a short seek-based decode at a higher rate, used to bisect counter jumps).
- `metadata.py`: `started_at(path, creation_time)`, the Recording's start from its file name, else the container tag, else `None`.
- `references.py` and `data/`: reference images shipped in the package (HUD icons, Victory Royale banner, `#` glyph).
- `screens.py`: pure per-frame detection, `hud_score`, `results_reading` and `counter_masks`; thresholds `HUD_THRESHOLD`/`RESULTS_THRESHOLD`.
- `ocr.py`: per-glyph Tesseract reading of a binary mask (`tesseract` CLI via subprocess).
- `counters.py`: `ReuseTracker` (the skip-OCR rule: read again only when the pixels differ from the last OCR'd ones), `CounterReader` (the serial reader built on it) and `CounterTracker` (turns readings into change Events and `unreadable` spans, and records jumps of more than one as `Gap`s for a resolver).
- `resample.py`: `resolve_jump` (pure: times each step of a jump from 4 fps samples) and `JumpResolver` (decodes each Gap's window, pooled per Match close). `analyse` seeks only for ambiguous gaps.
- `observations.py`: `SecondObservation` (everything read at one second), per-chunk extraction with packed counter masks, JSON (de)serialisation and the `--observations` records.
- `parallel.py`: the chunk plan and `spawn` process pool for frame facts (abort flag, error transport), and the OCR thread pool for counters.
- `detect.py`: `find_hud` and `find_results` locate the HUD elements in one frame (icons by multi-scale template match over the whole frame, counters and placements anchored on the OCR'd digits) and return normalised rectangles with their scores.
- `setup.py`: `run_setup(recording, at)` reads each `--at` second and its two neighbours, merges the best detection per element, builds the Layout and reads every element back with the `analyse` readers before accepting it; `preview.py` draws the annotated PNG.
- `manual.py` and `data/manual.html`: `setup --manual` serves a stdlib `ThreadingHTTPServer` on 127.0.0.1 under a random token path; the page posts `{elements: {name: {panel, x, y, w, h}}}` to `layout`, `setup.manual_result` validates it (hard errors are 400, read-back failures are warnings) and the CLI's `_write_layout` writes the files before the server shuts itself down.
- `selection.py`: `select_layout` drops Layouts whose aspect ratio does not fit the Recording, scores each remaining `hud_anchor` on up to 20 in-match frames (about 240 candidate seconds, plus at most 60 more from the dense opening window, read in time order so the choice does not depend on `--jobs`) and picks the best; none above `MIN_LAYOUT_SCORE` is a `LayoutSelectionError`. The first `NOT_FORTNITE_WINDOW` seconds are sampled at least every `OPENING_STRIDE` s; when none shows an in-match HUD and `detect.find_hud` finds no reference icons either, it returns `NoHud` (the `not_fortnite` warning) instead of raising.
- `cache.py`: Observation cache keyed by Recording fingerprint, Layout hash and `OBSERVATION_VERSION` (bump it when readings or thresholds change), plus the bisect windows. The selection scores live beside it in `selection-*.json`, keyed by the Recording and the set of Layout hashes and by both `OBSERVATION_VERSION` and `SELECTION_VERSION` (bump the latter when `SCORED_FRAMES`, `MAX_CANDIDATE_FRAMES` or how a score is formed changes).
- `analysis.py`: `analyse(path, layouts, jobs=, chunk_seconds=, cache=)` and `run_machine`, the state machine that turns per-second Observations into Matches and Events. A Match opens on the HUD and closes only on a results screen. Output must not depend on `jobs` or chunk size: the skip-OCR rule is replayed with `ReuseTracker` over the exact schedule of counter reads the machine produces, instead of running per chunk.

| Subcommand | Purpose | Status |
|------------|---------|--------|
| `setup` | Produce a Layout from a Recording | working: auto-detect from `--at hud=` and `--at results=`; `--manual` serves a local browser editor |
| `analyse` | Turn a Recording and a Layout into Matches with Events | working: Matches, Win, placement, HUD counters (speech: later tickets) |
| `review` | Inspect the Events found in a Recording | stub, see epic #13 |
| `schema` | Print the JSON schema of the Analysis output | working |

`review` is a stub: it prints `victorious review: not implemented yet` to stderr and exits 1. `setup` takes `RECORDING --at KIND=TIME ... -o LAYOUT [--preview PNG] [--manual [--no-browser] [--port N]]`; it exits 2 when a kind is missing, 1 when a required element is not found (it names them and points to `--manual`, writing no Layout). `analyse` takes `--jobs` (default: CPU count), `--cache-dir`/`--no-cache`, `--observations out.jsonl` and a hidden `--chunk-seconds`. Errors in `analyse` print `victorious analyse: <reason>` to stderr, exit 1 and write nothing.

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
