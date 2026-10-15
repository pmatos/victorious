# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victorious turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match (Wins, placement, eliminations, speech), read from what is on screen and what is said. The domain vocabulary (Recording, Match, Win, Layout, HUD element, Observation, Event, Setup, Analysis, Transcript segment, Music mask, Speaker, Verdict, Labels file) is defined in `GLOSSARY.md`; architecture decisions are in `docs/adr/`. Read both before changing behaviour. The work is tracked in epic #13.

## Architecture

Installable package in `src/victorious/` (hatchling build, console script `victorious`):

- `cli.py`: argparse entry point exposing the subcommands below.
- `__main__.py`: enables `python -m victorious`.
- `matcher.py`: internal multi-scale template matcher (scales 0.3-1.2) used as a detector for reference images such as the Victory Royale banner. Not a user-facing command.
- `layout.py`: pydantic `Layout` (YAML, normalized rects), `Layout.fits` (aspect ratio), `load_layout` and `load_layouts` (a file or a directory of Layouts, as `LoadedLayout`s).
- `schema.py`: pydantic output models (`AnalysisOutput`, `Match`, `Event`), `SCHEMA_VERSION`, `json_schema()`.
- `video.py`: `probe` (ffprobe, with the container `creation_time`, the number of audio streams, the codec and pixel format), `sample_audio` (one window of one audio stream as 16 kHz mono float32 through an ffmpeg pipe), `require_complete` (one tail seek that fails a truncated Recording), `sample_chunk` (one BGR frame per second of a whole-second chunk through an ffmpeg raw pipe) and `sample_window` (a short seek-based decode at a higher rate, used to bisect counter jumps). All three decoding functions and `frame_at` go through `_frames`: VAAPI when `resolve_hardware` finds a device whose frames equal software's byte for byte, software otherwise and for the rest of a chunk whose hardware decode fails part-way (`hwaccel.note` prints the one stderr line). The hardware filter ends `hwdownload,format=nv12,format=yuv420p` before the BGR conversion; without the last `format` the pixels differ from software's by up to 60.
- `hwaccel.py`: optional VAAPI decode. `candidates` (render devices from `VICTORIOUS_VAAPI_DEVICE` or `/dev/dri/renderD*`; the user's `LIBVA_DRIVER_NAME` first, then the vendor's drivers read from `/sys/class/drm`: `radeonsi`, `iHD`, `i965`), `environment` (the driver goes into the ffmpeg subprocess environment only), `resolve` (one probe per process and Recording kind, under a lock; `auto` is silent without a render device, notes a failure once), `Shared`/`share`/`adopt` (the parent's answer, mode and one-note flag handed to pool workers) and the mode (`VICTORIOUS_HWACCEL` `auto|off|vaapi`, hidden `analyse --hwaccel`; the tests run with `off` and the hardware tests skip when no device works). Output must not depend on it, so `OBSERVATION_VERSION` is not bumped.
- `metadata.py`: `started_at(path, creation_time)`, the Recording's start from its file name, else the container tag, else `None`.
- `references.py` and `data/`: reference images shipped in the package (HUD icons, Victory Royale banner, `#` glyph).
- `screens.py`: pure per-frame detection, `hud_score`, `results_reading` and `counter_masks`; thresholds `HUD_THRESHOLD`/`RESULTS_THRESHOLD`; the Victory Royale banner is matched at 0.9x-1.1x next to the Layout rect and, only when that misses, at 1.1x-1.6x (`BANNER_WIDE_SWEEP`) on a half-size copy of a `BANNER_WIDE_MARGIN` region, since it appears large and low before settling into the XP-screen size; `scale_range` gives the range read per element.
- `ocr.py`: per-glyph Tesseract reading of a binary mask (`tesseract` CLI via subprocess).
- `counters.py`: `ReuseTracker` (the skip-OCR rule: read again only when the pixels differ from the last OCR'd ones), `CounterReader` (the serial reader built on it), `lobby_filling` (whether `players_left` climbs like a filling lobby) and `CounterTracker` (turns readings into change Events and `unreadable` spans, and records jumps of more than one as `Gap`s for a resolver).
- `resample.py`: `resolve_jump` (pure: times each step of a jump from 4 fps samples) and `JumpResolver` (decodes each Gap's window, pooled per Match close). `analyse` seeks only for ambiguous gaps.
- `observations.py`: `SecondObservation` (everything read at one second), per-chunk extraction with packed counter masks, JSON (de)serialisation and the `--observations` records.
- `parallel.py`: the chunk plan and `spawn` process pool for frame facts (abort flag, error transport, the parent's hardware decode state), `gpu_slots` and the slot semaphore (with hardware decode at most `HW_SLOTS` = 2 chunks decode on the GPU at once, `VICTORIOUS_HW_SLOTS` overrides, 0 is software; a chunk that finds no free slot decodes in software and the last `jobs` chunks wait for a slot so the CPU does not outlast the GPU queue), and the OCR thread pool for counters.
- `detect.py`: `find_hud` and `find_results` locate the HUD elements in one frame (icons by multi-scale template match over the whole frame, counters and placements anchored on the OCR'd digits) and return normalised rectangles with their scores.
- `setup.py`: `run_setup(recording, at)` reads each `--at` second and its two neighbours (on a results screen up to `NEIGHBOUR_REACH` s either side while nothing is found or an element is at a size `analyse` does not read), merges the best detection per element (a readable size first, then the score), builds the Layout and reads every element back with the `analyse` readers before accepting it; `preview.py` draws the annotated PNG.
- `localpage.py`: `serve_until_saved(resources, post_route, save, port=, open_browser=, announce=)`, the stdlib `ThreadingHTTPServer` on 127.0.0.1 under a random token path shared by `setup --manual` and `review`: exact-route GETs from a dict (anything else is 404), one JSON POST route with a body limit, the save under a lock (`OSError` is a 500, a second save is 409), and a shutdown thread after the first 200.
- `manual.py` and `data/manual.html`: `setup --manual` serves the frames and state on `localpage`; the page posts `{elements: {name: {panel, x, y, w, h}}}` to `layout`, `setup.manual_result` validates it (hard errors are 400, read-back failures are warnings) and the CLI's `_write_layout` writes the files before the server shuts itself down.
- `labels.py`: the labels file (`Labels`, `LABELS_VERSION` "1.0", pydantic `extra="forbid"`, not part of the analysis schema), `labels_from`, `load_labels` and `compare(labels, output, tolerance)`. Matches pair with labels by span (`MIN_OVERLAP` 0.5 interval IoU, greedy by descending overlap), never by `index`; differences are `missing`, `changed`, `shifted`, `reappeared` and `unlabelled`.
- `review.py` and `data/review.html`: `load_analysis` (checks `schema_version` first), `extract_frames` (the start frame and the frame `RESULTS_SETTLE` s after the first results second of every Match, each one `video.frame_at` in a thread pool, JPEG), `match_cards` and `serve_review`, whose save closure needs a verdict for every Match (`{"verdicts": {"<index>": "confirmed"|"rejected"}}` posted to `labels`).
- `selection.py`: `select_layout` drops Layouts whose aspect ratio does not fit the Recording, scores each remaining `hud_anchor` on up to 20 in-match frames (about 240 candidate seconds, plus at most 60 more from the dense opening window, read in time order so the choice does not depend on `--jobs`) and picks the best; none above `MIN_LAYOUT_SCORE` is a `LayoutSelectionError`. The first `NOT_FORTNITE_WINDOW` seconds are sampled at least every `OPENING_STRIDE` s; when none shows an in-match HUD and `detect.find_hud` finds no reference icons either, it returns `NoHud` (the `not_fortnite` warning) instead of raising. When no Layout fits the aspect ratio, the same icon scan decides between `NoHud` and the no-fit `LayoutSelectionError`.
- `cache.py`: Observation cache keyed by Recording fingerprint, Layout hash and `OBSERVATION_VERSION` (bump it when readings or thresholds change), plus the bisect windows. The selection scores live beside it in `selection-*.json`, keyed by the Recording and the set of Layout hashes and by both `OBSERVATION_VERSION` and `SELECTION_VERSION` (bump the latter when `SCORED_FRAMES`, `MAX_CANDIDATE_FRAMES`, `NOT_FORTNITE_WINDOW`, `OPENING_STRIDE` or how a score is formed changes).
- `speech.py`: `transcribe_matches` (per Match window: ffmpeg audio, faster-whisper VAD, per-chunk language over `en`/`pt` (both decoded when detection is unsure, `pick_reading` keeps the more confident), `is_speech` drop rule and `drop_repeated` loop filter, nearest-Match ownership of margin segments, merge into the Match's Events without moving existing ones), `require_ready` (Hugging Face token present, audio tagger and diarization model loadable, `voices/` clips decodable) and `require_audio_stream`. `faster_whisper` is imported only when the model is loaded, so `--no-speech` never imports it.
- `speakers.py`: `label_segments` (one diarization per Recording over the padded, joined audio around the segments: `padded_spans`, `pieces_for`/`clamped`, `speaker_for_segment` max overlap, `name_speakers` nearest enrolled within `MATCH_DISTANCE` else `unknown_N` by first appearance), `load_clips` (enrollment folder, stems validated), `fetch_diarizer` (pinned ungated mirror, local cache first) and `Diarizer` (pyannote pipeline and WeSpeaker embedder; `torch` and `pyannote` are imported only when it loads, so `--no-speech` never imports them). See ADR 0004.
- `music.py`: `MusicTagger` (YAMNet on `onnxruntime`, per-second `Music`/`Speech` scores), `masked_windows` (music-dominant runs: threshold, one-second bridge, speech-adjacent seconds trimmed, 3 s minimum, one-second edges), `merge_windows`, `unmasked` and `silence`. `transcribe_matches` tags the merged Match windows first (one decision per Recording second); `_transcribe_window` finds VAD spans on the unmodified audio, cuts them at the masked seconds (`unmasked`) and silences those samples before language detection and Whisper, so output outside masks is unchanged. `onnxruntime` is imported only when the tagger loads, so `--no-speech` never imports it.
- `analysis.py`: `analyse(path, layouts, jobs=, chunk_seconds=, cache=, speech=)` and `run_machine`, the state machine that turns per-second Observations into Matches and Events. A Match opens on the HUD and closes on a results screen, or, when the HUD returns after more than 20 s with `lobby_filling` true over the next 6 s of readings, at its last HUD second (an abandoned Match), unless its HUD run was under 30 s and only a filling lobby, which is not a Match and is dropped. Output must not depend on `jobs` or chunk size: the skip-OCR rule is replayed with `ReuseTracker` over the exact schedule of counter reads the machine produces, instead of running per chunk, so that schedule (resets and reads) must not depend on counter values.

| Subcommand | Purpose | Status |
|------------|---------|--------|
| `setup` | Produce a Layout from a Recording | working: auto-detect from `--at hud=` and `--at results=`; `--manual` serves a local browser editor |
| `analyse` | Turn a Recording and a Layout into Matches with Events | working: Matches, Win, placement, HUD counters, speech transcripts with music masked and Speakers (categories: later ticket) |
| `review` | Confirm or reject the detected Matches into a labels file; `--against` compares a run with one | working |
| `schema` | Print the JSON schema of the Analysis output | working |

`review` takes `ANALYSIS_JSON [--recording PATH] [-o LABELS] [--no-browser] [--port N]` to serve the page and write the labels (exit 1 when cancelled), or `ANALYSIS_JSON --against LABELS [--tolerance SECONDS]` to print the differences (exit 1 when there are any or the labels are for another Recording); combining `--against` with the serving options exits 2. `setup` takes `RECORDING --at KIND=TIME ... -o LAYOUT [--preview PNG] [--manual [--no-browser] [--port N]]`; it exits 2 when a kind is missing, 1 when a required element is not found (it names them and points to `--manual`, writing no Layout). `analyse` takes `--jobs` (default: CPU count), `--cache-dir`/`--no-cache`, `--observations out.jsonl`, `--audio-stream N`, `--voices DIR` (enrollment clips, default `./voices`; speaker tests pass it or run in a temp directory), `--no-speech` (default on when `VICTORIOUS_NO_SPEECH` is set; the tests set it) and hidden `--chunk-seconds`, `--whisper-model` and `--diarization-model`. Without `--no-speech` a Hugging Face token must be set or it exits 1 before any decode. Errors in `analyse` print `victorious analyse: <reason>` to stderr, exit 1 and write nothing.

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

Plain `uv run pytest` runs every test. Tests that load the real Whisper, YAMNet or pyannote models (every test that uses a `speech_on` fixture; `tests/conftest.py` marks them) carry the `slow_models` marker, and `uv run pytest -m "not slow_models"` skips them for a quick local loop. CI (`.github/workflows/ci.yml`) runs once per push: on `main`, nightly (03:17 UTC) and on `workflow_dispatch` it runs everything, and on a pull request it adds `-m "not slow_models"` unless the diff touches a path listed in `.github/slow-models-paths.txt` (checked by `.github/scripts/slow-models-needed.sh`). A new test module that turns speech on needs its file in that list (`tests/test_ci_config.py` checks this). The pytest step prints the 25 slowest tests.

## Test fixtures

Tests build their Recordings on demand with `tests.synthetic`; no video is committed. `tests/fixtures/speech/` holds two short synthetic speech clips (`make_speech.py` regenerates them). `tests/test_speech.py` and `tests/test_music_masking.py` run the real `small` Whisper model and the real YAMNet ONNX model, so a cold Hugging Face cache needs network once; the music bed is synthesised at test time (`tests/synthetic/music.py`, chords, drums and a sung vowel line that Whisper reads as "You"), nothing binary is committed; `conftest.py` pins `HF_HOME` so the isolated XDG cache does not hide it.

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
