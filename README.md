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
| `analyse` | Turn a Recording and a Layout into Matches with Events | working (Matches, Win, final placement, HUD counters) |
| `review` | Inspect the Events found in a Recording | placeholder |
| `schema` | Print the JSON schema of the Analysis output | working |

### Analyse

```bash
uv run victorious analyse recording.mkv --layout layout.yaml -o analysis.json
uv run victorious schema -o analysis.schema.json
```

One Recording is analysed per invocation; JSON goes to stdout unless `-o` is given, and the file is only written when the whole run succeeded. The Recording is sampled once per second, so every time is in whole seconds from the start of the Recording, except counter Events resolved from a multi-step jump (see below), which are on a 0.25 s grid.

A Match starts when the in-match HUD appears and ends at its results screen; being eliminated, or losing the HUD while spectating, does not end it. A Match still running when the Recording ends is reported with `win: false` and `placement: null`. A results screen whose placement cannot be read also gives `placement: null`. `analyse` refuses a Recording whose aspect ratio differs (beyond 2%) from the Layout's `aspect_ratio`.

The output carries `schema_version` and, per Match, `start`, `end`, `win`, `placement`, `personal_eliminations` and its Events. Every Event has `kind`, `time`, `source`, `confidence`, `value`, `end` and `element`.

- `match_start`, `win`, `final_placement` (`value` is the placement) and `match_end` come from the Match boundaries and the results screen.
- `personal_eliminations`, `team_eliminations` and `players_left` are HUD counters, read with Tesseract once per sampled second. An Event is emitted only when the value changes, with the new count in `value` and the first second it was read. A value is committed only after it is read in two consecutive readable seconds, so one misread frame never becomes an Event, and the first value read in a Match is a silent baseline, not an Event. When a counter moves by more than one between two sampled seconds, only that gap is decoded again at 4 fps and every step gets its own Event, timed to the 0.25 s slot in which it appears (never later than the change, at most 0.25 s earlier); steps that fall in the same slot share its time, and the number of Events equals the number of steps. This applies to rises of up to 3 in `personal_eliminations` and `team_eliminations` and falls of up to 10 in `players_left`; larger or wrong-direction jumps are OCR noise candidates and stay one Event. A gap is not re-sampled when a second inside it was unreadable, when it spans more than 3 s, or after an `unreadable` span: those jumps stay one Event at the first second the new value was read. Single-step changes cause no extra decoding. Values are never interpolated.
- `unreadable` marks a counter (`element`) with no usable reading for more than 10 s, from `time` to `end` (a span of exactly 10 s is not reported). It has `confidence: 0` and `value: null`.
- `Match.personal_eliminations` is the last committed personal eliminations count, or `null` when it was never read or the Layout has no `personal_eliminations` rect. A counter missing from the Layout is not read at all, so it produces neither Events nor `unreadable`.

Counter digits are about 17 px tall at 2560x1440 and Tesseract reads them reliably from 1920x1080 up; at 720p they are unreliable.

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
    elimination_placement: {x: 0.504, y: 0.104, w: 0.125, h: 0.111}
```

`hud_anchor` must cover the three icons (skull, squad, person) stacked in the HUD column; `results_banner` and `placement` locate the Victory Royale banner (blue or orange) and the gold `#N` that goes with it. `elimination_placement`, which is optional, locates the white `YOU PLACED #N` shown when the player is eliminated; it is centred on the frame, so it spans the whole width the `#` can land in. Without it, only Wins end a Match. The three counters are optional and read by later tickets.

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
