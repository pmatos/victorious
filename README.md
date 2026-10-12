# Victorious

Turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match: Wins, final placement, eliminations and speech.

`setup`, `analyse` and `schema` work today. `review` is a placeholder that prints a not-implemented message and exits 1 until its ticket in [epic #13](https://github.com/pmatos/victorious/issues/13) lands.

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
| `setup` | Produce a Layout from a Recording | working (auto-detect, or `--manual` in a local browser page) |
| `analyse` | Turn a Recording and a Layout into Matches with Events | working (Matches, Win, final placement, HUD counters) |
| `review` | Inspect the Events found in a Recording | placeholder |
| `schema` | Print the JSON schema of the Analysis output | working |

### Setup

```bash
uv run victorious setup recording.mkv --at hud=12:34 --at results=1:05:20 -o layout.yaml
```

Point `setup` at one second that shows the in-match HUD and one that shows a results screen. It finds the three HUD icons and the Victory Royale banner from the reference images that ship with the tool, anchors the counters and the placement on the digits it reads next to them, and writes a Layout `analyse` accepts. A second `layout.preview.png` (beside the Layout, or `--preview PATH`) shows each reference frame with every rectangle drawn and labelled, stacked in the order of the `--at` options, so a glance tells whether they sit where they should.

| Option | Meaning |
|--------|---------|
| `--at hud=TIME`, `--at results=TIME` | required, once per kind at least; `TIME` is seconds (`754`, `754.5`) or `MM:SS` / `H:MM:SS`, floored to a whole second |
| `-o`, `--output PATH` | where the Layout (YAML) goes; required, an existing file is replaced |
| `--preview PATH` | where the preview PNG goes (default: `<output stem>.preview.png`) |
| `--manual` | place the elements by hand on a local browser page, pre-filled with what auto-detect found |
| `--no-browser` | with `--manual`, print the address and do not open a browser |
| `--port N` | with `--manual`, listen on port `N` of `127.0.0.1` (default: any free port) |

Either kind may be repeated, and the best match of each element wins. A results frame shows only one style: a Win has the banner and the gold `#N` (`results_banner`, `placement`), an eliminated player sees a white `YOU PLACED #N` (`elimination_placement`). Pass one of each to get all three; without an eliminated frame the Layout is still written and a warning on stderr names `elimination_placement`. A counter that cannot be read on a frame is reported by name: try another `--at hud=` second where the digits are crisp. The seconds either side of each `--at` are also tried, so one unlucky video frame does not fail a counter.

When a required element cannot be found, `setup` exits 1, names each one with the reason, points to `--manual` and writes no Layout (the preview is still written when anything was found). A missing `--at` kind, or a time that is not a time, is a usage error (exit 2). `setup` finds HUDs between 0.7x and 1.4x of the reference size, but `analyse` reads only 0.9x to 1.1x, so a Layout that would not read back is reported instead of written. The Victory Royale banner animates: pick a second just after it appears, while it is full size, since on some XP screens it shrinks below 0.9x and `setup` then reports it.

#### Placing the elements by hand

```bash
uv run victorious setup recording.mkv --at hud=12:34 --at results=1:05:20 --manual -o layout.yaml
```

`--manual` runs auto-detect first, then serves a page on `http://127.0.0.1:PORT/TOKEN/` (the address is printed on stderr and opened in your browser unless `--no-browser`; the random token keeps other pages and users from posting to it). The page shows the frame of every `--at`; the rectangles auto-detect found are already drawn and each element it did not find says why. Pick an element from the fixed list, drag on a frame of the matching kind to draw it, drag a box to move it, its corner to resize it, or type exact values (fractions of the frame, so the resolution does not matter). `Save Layout` writes the same YAML Layout and preview as auto-detect, then the server exits 0. `Ctrl-C` cancels: exit 1, nothing written.

Saving is refused, with the reasons shown on the page, when a required element is missing, an element sits on a frame of the wrong kind or a rectangle leaves the frame. A Layout that is complete but does not read back with the `analyse` readers is still written, because you are the authority; each element that fails is a warning on the page and on stderr.

### Analyse

```bash
uv run victorious analyse recording.mkv --layout layout.yaml -o analysis.json
uv run victorious schema -o analysis.schema.json
```

One Recording is analysed per invocation; JSON goes to stdout unless `-o` is given, and the file is only written when the whole run succeeded. The Recording is sampled once per second, so every time is in whole seconds from the start of the Recording, except counter Events resolved from a multi-step jump (see below), which are on a 0.25 s grid.

A Match starts when the in-match HUD appears and ends at its results screen; being eliminated, or losing the HUD while spectating, does not end it. A Match still running when the Recording ends is reported with `win: false` and `placement: null`. A results screen whose placement cannot be read also gives `placement: null`.

The output carries `schema_version` (1.4), `recording` (`path`, `duration` and `started_at`, see below), `layout` (the Layout used and its score, see below), `warnings` and, per Match, `start`, `end`, `win`, `placement`, `personal_eliminations` and its Events. Every Event has `kind`, `time`, `source`, `confidence`, `value`, `end`, `element`, `text` and `language`.

- `match_start`, `win`, `final_placement` (`value` is the placement) and `match_end` come from the Match boundaries and the results screen.
- `personal_eliminations`, `team_eliminations` and `players_left` are HUD counters, read with Tesseract once per sampled second. An Event is emitted only when the value changes, with the new count in `value` and the first second it was read. A value is committed only after it is read in two consecutive readable seconds, so one misread frame never becomes an Event, and the first value read in a Match is a silent baseline, not an Event. When a counter moves by more than one between two sampled seconds, only that gap is decoded again at 4 fps and every step gets its own Event, timed to the first readable 0.25 s slot in which it appears (an unreadable or misread sample inside the window can delay it by a slot); steps that fall in the same slot share its time, and the number of Events equals the number of steps. This applies to rises of up to 3 in `personal_eliminations` and `team_eliminations` and falls of up to 10 in `players_left`; larger or wrong-direction jumps are OCR noise candidates and stay one Event. A gap is not re-sampled when a second inside it was unreadable, when it spans more than 3 s, or after an `unreadable` span: those jumps stay one Event at the first second the new value was read. The eliminations counters never fall within a Match, so a rise only gets steps beyond the highest value already committed; a recovery to that value after a sustained misread stays one Event. Single-step changes cause no extra decoding. Values are never interpolated.
- `unreadable` marks a counter (`element`) with no usable reading for more than 10 s, from `time` to `end` (a span of exactly 10 s is not reported). It has `confidence: 0` and `value: null`.
- `Match.personal_eliminations` is the last committed personal eliminations count, or `null` when it was never read or the Layout has no `personal_eliminations` rect. A counter missing from the Layout is not read at all, so it produces neither Events nor `unreadable`.

- `transcript` is one segment of speech (`source: "speech"`), with `text`, `language` (`en` or `pt`) and `confidence` (`exp(avg_logprob)` of the segment); `time` and `end` are its true span in seconds from the start of the Recording, so a call just after the results screen can lie beyond `Match.end`. Events of a Match are in time order, which means a segment spoken before the HUD appeared precedes `match_start`.

`recording.started_at` is an ISO 8601 time. It is read from the file name when that holds `YYYY-MM-DD_HH-MM-SS` or `YYYY-MM-DD HH-MM-SS` (OBS's names; local time, so no offset: `2026-10-09T11:59:53`), else from the container's `creation_time` tag (with its UTC offset: `2026-10-09T09:59:53+00:00`), else it is `null`. An impossible date in the name falls through to the container, then `null`.

A Recording that is cut short is an error, not a partial result: `analyse` exits 1, writes no output (and no `--observations` file) and names the file, when the header promises a duration that the decodable video falls more than 2 s short of, or when its header or first frame cannot be read.

Counter digits are about 17 px tall at 2560x1440 and Tesseract reads them reliably from 1920x1080 up; at 720p they are unreliable.

#### Layout selection

`--layout` takes one Layout file or a directory of them (`*.yaml` and `*.yml` directly inside it, no recursion; an invalid file is an error naming it). Layouts whose `aspect_ratio` differs from the Recording's by more than 2% are dropped; if none is left, `analyse` exits 1 with an error that names the Layouts' ratios, the Recording's `WxH` and `victorious setup`. Each remaining Layout's `hud_anchor` is then scored on in-match frames: one frame every `ceil(duration / 240)` s (at least 1 s; at most every 10 s in the first 10 minutes, so about 240 candidates plus at most 60) from the start is read in time order until 20 of them show an in-match HUD (some candidate Layout scores at least 0.7), and a Layout's score is its mean over those 20. The best score wins, ties go to the first file by name, and the choice does not depend on `--jobs`.

The output records `layout: {path, score}`, with `path` as given on the command line. When no Layout reaches the minimum score of 0.7, or no sampled frame scores at the anchor of any Layout although the HUD icons are on screen (a HUD no Layout describes), `analyse` exits 1, writes no output and says `no Layout matches`, naming `victorious setup`.

A Recording in which no in-match HUD appears in the first 10 minutes, for any Layout and to the HUD detector `setup` uses (the reference icons), is not Fortnite: `analyse` exits 0 with `matches: []`, `layout: null` and `warnings: [{code: "not_fortnite", message}]`, and repeats the warning on stderr. `--observations` then writes an empty file. A HUD that first appears after minute 10 is not looked for. Such a result is not cached, but only a complete Recording gets it: a truncated one is the error above. Scores are cached next to the Observations, keyed by the Recording and the set of Layout contents, so a rerun does no selection decoding; `--no-cache` disables this too. A Recording whose HUD changes design part-way averages low for every Layout and is refused.

#### Parallel decode, cache and `--observations`

`analyse` reads the Recording in whole-second chunks (60 s) on up to `--jobs` processes (default: one per CPU), each chunk decoded by its own ffmpeg. Chunk boundaries sit on whole seconds, so every chunk samples exactly the frames a single serial decode would, and the counter skip-OCR rule is replayed over the Match loop's exact schedule afterwards; the output is byte-identical for any `--jobs` and chunk size. A Recording whose chunks do not join up (missing seconds) is rejected as undecodable.

The per-second Observations are cached, keyed by a fingerprint of the Recording (its size and its first, middle and last MiB, not its name or mtime) and a hash of the Layout (its content, not its YAML formatting). A rerun with the same Recording and Layout loads them, and the windows that time counter jumps, so it runs no ffmpeg at all. Changing any Layout rectangle or the aspect ratio, or replacing the Recording, misses the cache; an unreadable entry is recomputed. The cache lives in `$XDG_CACHE_HOME/victorious` (`~/.cache/victorious`) unless `--cache-dir` says otherwise, `--no-cache` neither reads nor writes it, and nothing is ever evicted: an entry is about 3-4 MB per 3 h Recording per Layout. A failed run stores no Observations, and a cache that cannot be written is skipped, not an error. A line on stderr reports the Layout chosen with its score and the number of candidates, `cache hit`, `cache miss` or `cache off`, and the wall time.

`--observations observations.jsonl` also writes every raw reading, one JSON line per HUD element per sampled second, in the order `hud_anchor`, `personal_eliminations`, `team_eliminations`, `players_left`, `results_banner`, `placement`, `elimination_placement` (elements the Layout does not describe are left out):

| Field | Meaning |
|-------|---------|
| `second` | whole seconds from the start of the Recording |
| `element` | the HUD element |
| `value` | the digits read as an integer, `null` when none were read; only counters and placements have one |
| `confidence` | OCR confidence of `value` (0..1), `0` when `value` is `null` |
| `score` | template match score for `hud_anchor`, `results_banner`, `placement` and `elimination_placement`; `null` for counters |
| `region` | the Layout's `{x, y, w, h}` for the element |

Counters are read only while a Match is open and the HUD is on screen, and only when their pixels changed; any other second has `value: null` for them, and a second that reused the pixels of an earlier one carries that earlier reading. `placement` and `elimination_placement` carry the digits read on the results screen, `value` going to whichever of the two "#" styles matched best. When speech ran and music was masked, one more line per masked span follows the HUD records, `{"element": "music_mask", "start": s, "end": s, "music": m, "speech": p}` (seconds from the start of the Recording; `music` and `speech` are the span's mean per-second scores). The file is written, like `-o`, only after the whole run succeeded; it is not part of the exported JSON schema.

#### Speech

Unless `--no-speech` is given, `analyse` transcribes the audio of every Match window, `[start - 10 s, end + 10 s]` clamped to the Recording, with faster-whisper (model `small`, int8 on CPU, batched). The first audio stream is used; `--audio-stream N` picks another one (counted from 0 among the audio streams). The language is detected per stretch of speech and restricted to English and Portuguese, because Recordings switch between the two; when detection is unsure (the two are within 0.5 of each other), the stretch is decoded in both and the reading with the higher average confidence is kept. Segments whose `no_speech_prob` exceeds 0.9, whose `avg_logprob` is below -1.5 or whose compression ratio exceeds 2.4 are dropped, and so is any run of three or more back-to-back segments with the same words (Whisper's loops on music and noise), so silence and game sound yield no Events. A segment in the margin shared by two Matches goes to the Match whose span is nearer to the middle of the segment (the earlier one on a tie); nothing is duplicated. Audio outside every Match window is never transcribed.

Before transcription, music-dominant audio is silenced so that emote and lobby music is not read as lyrics. A YAMNet audio tagger (ONNX, CPU, about 300 times faster than real time) scores each second of every Match window for `Music` (mean over the frames in that second) and `Speech` (max): a second is music-dominant when music is at least 0.5 and speech below 0.25, so a second with speech over music is never masked. A run of music-dominant seconds bridges one second where the music dipped but nobody spoke, gives up its first or last second when speech is heard right next to it (speech starts and ends inside the second the tagger still calls music), takes in the second on each side where music is still at least 0.25 and nobody speaks, and is masked when what is left lasts at least 3 s (stingers and single beats are left alone). Speech spans are found on the unmodified window audio and then cut at the masked seconds, so a window without a mask is transcribed exactly as before and transcripts outside masks do not move. Overlapping Match margins are tagged once, so a Recording second gets one decision. The masked spans are written to `--observations` as `music_mask` lines and counted on the stderr line (`music masked N s`). The tagger (about 16 MB, pinned to a revision of `audiomagic/yamnet-onnx`) is public and downloaded on first use into the Hugging Face cache. Known limit: vocal music that the tagger scores as `Speech` is not masked and is still transcribed.

Speech needs a Hugging Face token (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or `hf auth login`); without one `analyse` exits 1 before reading the Recording and says so. The token is only checked for presence here. The Whisper model is public: it is downloaded on first use into the Hugging Face cache (about 480 MB) and loaded from it afterwards. A Recording without audio, or an `--audio-stream` it does not have, is an error that names `--no-speech`.

`--no-speech` skips all of this (setting `VICTORIOUS_NO_SPEECH` to `1`, `true` or `yes` makes it the default): no token is needed and no speech library is imported. Transcripts are not part of the Observation cache; a cached rerun still transcribes.

### Layout file

`setup` writes a Layout for you; one can also be written by hand. Keep several in one directory and pass the directory to `--layout`. Coordinates are fractions of the frame (`x`, `y`, `w`, `h` in 0..1), so one Layout serves every resolution of the same HUD design:

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
