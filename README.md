# Victorious

Turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match: Wins, final placement, eliminations and speech.

`analyse` and `schema` work today. `setup` and `review` are placeholders that print a not-implemented message and exit 1 until their tickets in [epic #13](https://github.com/pmatos/victorious/issues/13) land.

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
| `analyse` | Turn a Recording and a Layout into Matches with Events | working (Matches, Win, final placement) |
| `review` | Inspect the Events found in a Recording | placeholder |
| `schema` | Print the JSON schema of the Analysis output | working |

### Analyse

```bash
uv run victorious analyse recording.mkv --layout layout.yaml -o analysis.json
uv run victorious schema -o analysis.schema.json
```

One Recording is analysed per invocation; JSON goes to stdout unless `-o` is given, and the file is only written when the whole run succeeded. The Recording is sampled once per second, so every time is in whole seconds from the start of the Recording.

A Match starts when the in-match HUD appears and ends at its results screen; being eliminated, or losing the HUD while spectating, does not end it. A Match still running when the Recording ends is reported with `win: false` and `placement: null`. A results screen whose placement cannot be read also gives `placement: null`. `analyse` refuses a Recording whose aspect ratio differs (beyond 2%) from the Layout's `aspect_ratio`.

The output carries `schema_version` and, per Match, `start`, `end`, `win`, `placement` and its Events (`match_start`, `win`, `final_placement`, `match_end`), each with `time`, `source` and `confidence`.

### Layout file

Until `setup` exists, a Layout is written by hand. Coordinates are fractions of the frame (`x`, `y`, `w`, `h` in 0..1), so one Layout serves every resolution of the same HUD design:

```yaml
aspect_ratio: "16:9"
screens:
  hud:
    hud_anchor: {x: 0.852, y: 0.030, w: 0.014, h: 0.088}
    personal_eliminations: {x: 0.822, y: 0.037, w: 0.027, h: 0.012}
    team_eliminations: {x: 0.822, y: 0.065, w: 0.027, h: 0.012}
    players_left: {x: 0.822, y: 0.099, w: 0.027, h: 0.012}
  results:
    results_banner: {x: 0.432, y: 0.076, w: 0.148, h: 0.132}
    placement: {x: 0.359, y: 0.104, w: 0.109, h: 0.097}
```

`hud_anchor` must cover the three icons (skull, squad, person) stacked in the HUD column; `results_banner` and `placement` locate the Victory Royale banner and the `#N` placement. The three counters are optional and read by later tickets.

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
- OpenCV (opencv-python), NumPy, pydantic, PyYAML
- `ffmpeg` and `ffprobe` on `PATH` (decoding)
- `tesseract` on `PATH` (reading placement digits), e.g. `apt install tesseract-ocr`
