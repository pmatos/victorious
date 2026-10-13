# Victorious

Turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match: Wins, final placement, eliminations and speech.

All four subcommands work. The remaining work is tracked in [epic #13](https://github.com/pmatos/victorious/issues/13).

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Tutorial

From a Recording to a reviewed result. The commands below use `recording.mkv`; the reference sections further down say what every option does.

You need `ffmpeg`, `ffprobe` and `tesseract` on `PATH` (see [Requirements](#requirements)) and `uv sync` done.

### 1. Set up a Layout (once per HUD design and aspect ratio)

Find, in a video player, one second where the in-match HUD is on screen (the three counters at the top right) and one second of each results screen: a Win (Victory Royale banner) and a "YOU PLACED #N" screen. Then:

```bash
uv run victorious setup recording.mkv --at hud=12:34 --at results=1:05:20 --at results=1:41:07 -o layout.yaml
```

`TIME` is seconds (`754`) or `MM:SS` / `H:MM:SS`. This writes `layout.yaml` and `layout.preview.png`. Open the preview: every element is drawn and labelled on each reference frame, so a rectangle in the wrong place is obvious.

- With only a Win frame, the Layout is still written and stderr warns that `elimination_placement` was not found. Add a `--at results=` second showing "YOU PLACED #N" and rerun.
- `could not find N element`: try another `--at hud=` second where the digits are crisp, or add `--manual` to drag the rectangles into place on a page served on `127.0.0.1` (pre-filled with what was found; add `--no-browser` to only print its address; `Save Layout` writes the same files).

### 2. Analyse

```bash
uv run victorious analyse recording.mkv --layout layout.yaml --no-speech -o analysis.json
```

Start with `--no-speech`: it needs no token and no models, and gives the Matches, Wins, placement and counters. `--layout` also takes a directory of Layouts (one per HUD design or aspect ratio); the best fit for the Recording is chosen. Without `-o` the JSON goes to stdout. `--observations observations.jsonl` also writes every raw HUD reading, one line per element per second, which is what to read when a counter looks wrong.

Reruns on the same Recording and Layout reuse the cached Observations (`--no-cache` to ignore them, `--jobs N` to limit the processes).

### 3. Read the output

Every time is in seconds from the start of the Recording.

```bash
jq -c '.matches[] | {start, end, win, placement, personal_eliminations}' analysis.json
```

```text
{"start":2.0,"end":20.0,"win":true,"placement":1,"personal_eliminations":1}
{"start":30.0,"end":45.0,"win":false,"placement":6,"personal_eliminations":0}
```

Each Match also has an `events` list in time order: `match_start`, `personal_eliminations` and `team_eliminations` (with the new count in `value`), `players_left`, `win`, `final_placement` (`value` is the placement), `match_end` and, with speech, `transcript`. `placement: null` means the Match was still running at the end of the Recording or its placement could not be read. A `warnings` entry of `not_fortnite` (see [Common errors](#common-errors)) means no in-match HUD was found, so `matches` is empty.

```bash
uv run victorious schema -o analysis.schema.json
```

`schema` prints the JSON schema of this output (to stdout without `-o`), for validating or generating types.

### 4. Add speech (transcripts, Speakers, categories)

Speech is on unless `--no-speech` is given, and needs three things set up once:

1. **Hugging Face token.** Export `HF_TOKEN` (or `HUGGING_FACE_HUB_TOKEN`, or run `hf auth login`). Without one, `analyse` exits 1 before reading the Recording. The models are downloaded on first use (Whisper is about 480 MB).
2. **Enrolled voices (optional).** Put one clip of a few seconds per person in a `voices/` folder in the directory you run the tool from, named by the person: `voices/alice.wav` becomes the Speaker `alice`. Any other voice is `unknown_1`, `unknown_2`, ... Use `--voices DIR` for another folder.
3. **Jev API key (optional).** `export TYPESAFE_API_KEY=...` makes each transcript `category` `personal` or `game_related`. Only the text of each segment is sent to the external service (see [ADR 0002](docs/adr/0002-external-categorization-of-transcript-text.md)). Without a key a note goes to stderr; segments Jev cannot judge are `null` too. Either way the run still exits 0.

```bash
export HF_TOKEN=hf_...
uv run victorious analyse recording.mkv --layout layout.yaml -o analysis.json
```

The first audio stream is transcribed; if the voices (microphone, party chat) are on another track, pass `--audio-stream N` (counted from 0 among the audio streams). Speech takes a large share of the run time, so keep `--no-speech` (or `VICTORIOUS_NO_SPEECH=1`) while iterating on a Layout.

### 5. Review the Matches

```bash
uv run victorious review analysis.json
```

A page opens (the address is printed on stderr; `--no-browser` only prints it) with one card per Match and two frames of the Recording. Confirm a Match when what was read is right, reject it when it is not a real Match or a value is wrong (`c` and `r` decide the focused card). `Save labels` writes `analysis.labels.json` beside the JSON and the command exits 0; `Ctrl-C` cancels and writes nothing. If the Recording moved, pass `--recording PATH`.

Keep the labels file beside the Recording. After changing the tool or the Layout, rerun `analyse` and compare:

```bash
uv run victorious review analysis.json --against analysis.labels.json
```

It prints one line per difference (`missing`, `changed`, `shifted`, `reappeared`, `unlabelled`) and exits 1, or prints `no differences (N matches)` and exits 0. `--tolerance SECONDS` sets how far a boundary may move (default 2).

### Common errors

| Message or symptom | Cause and fix |
|--------------------|---------------|
| `no Layout fits this 1920x1080 Recording (layout.yaml is 4:3); run victorious setup ...` | Every Layout's aspect ratio differs from the Recording's by more than 2% and the Recording shows the in-match HUD (one with no HUD is reported as `not_fortnite` instead). Run `setup` on a Recording with this aspect ratio and put the result in the `--layout` directory. |
| `no Layout matches this Recording's HUD: ...` | The aspect ratio fits but no HUD anchor is found in the sampled frames: the HUD design differs or the rectangles are off. Check the preview, or rerun `setup` with other `--at` seconds. |
| `warning: not_fortnite: no in-match HUD in the first 10 minutes ...` (exit 0, `matches: []`, `layout: null`) | No in-match HUD was seen in the first 10 minutes for any Layout: not a Fortnite Recording, or a HUD design no Layout describes. A HUD that first appears after minute 10 is not looked for. |
| `Recording not found: FILE` or `cannot read Recording FILE: ...` (exit 1) | The file is missing or not a video. |
| `cannot read Recording FILE: it is truncated (no frame at 83 s, its header claims 85 s)` (exit 1) | A cut-short Recording is an error, not a partial result, and nothing is written. Check that it plays to the end and is not still being written. |
| `no Hugging Face token found: set HF_TOKEN ...` (exit 1) | Speech is on but there is no token. Set `HF_TOKEN`, or pass `--no-speech`. |
| `setup: could not find N element` (exit 1) | No Layout was written. Pass other `--at` seconds with a clear HUD or results screen, or use `--manual`. |
| Counters missing or wrong at 720p | Digits are only reliable from 1920x1080 up; use `--observations` to see what was read. |

## Usage

```bash
uv run victorious --help
```

| Subcommand | Purpose | Status |
|------------|---------|--------|
| `setup` | Produce a Layout from a Recording | working (auto-detect, or `--manual` in a local browser page) |
| `analyse` | Turn a Recording and a Layout into Matches with Events | working (Matches, Win, final placement, HUD counters, speech with Speakers) |
| `review` | Confirm or reject the detected Matches into a labels file, or compare a run against one | working |
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

The output carries `schema_version` (1.6), `recording` (`path`, `duration` and `started_at`, see below), `layout` (the Layout used and its score, see below), `warnings` and, per Match, `start`, `end`, `win`, `placement`, `personal_eliminations` and its Events. Every Event has `kind`, `time`, `source`, `confidence`, `value`, `end`, `element`, `text`, `language`, `speaker`, `category` and `category_confidence`.

- `match_start`, `win`, `final_placement` (`value` is the placement) and `match_end` come from the Match boundaries and the results screen.
- `personal_eliminations`, `team_eliminations` and `players_left` are HUD counters, read with Tesseract once per sampled second. An Event is emitted only when the value changes, with the new count in `value` and the first second it was read. A value is committed only after it is read in two consecutive readable seconds, so one misread frame never becomes an Event, and the first value read in a Match is a silent baseline, not an Event. When a counter moves by more than one between two sampled seconds, only that gap is decoded again at 4 fps and every step gets its own Event, timed to the first readable 0.25 s slot in which it appears (an unreadable or misread sample inside the window can delay it by a slot); steps that fall in the same slot share its time, and the number of Events equals the number of steps. This applies to rises of up to 3 in `personal_eliminations` and `team_eliminations` and falls of up to 10 in `players_left`; larger or wrong-direction jumps are OCR noise candidates and stay one Event. A gap is not re-sampled when a second inside it was unreadable, when it spans more than 3 s, or after an `unreadable` span: those jumps stay one Event at the first second the new value was read. The eliminations counters never fall within a Match, so a rise only gets steps beyond the highest value already committed; a recovery to that value after a sustained misread stays one Event. Single-step changes cause no extra decoding. Values are never interpolated.
- `unreadable` marks a counter (`element`) with no usable reading for more than 10 s, from `time` to `end` (a span of exactly 10 s is not reported). It has `confidence: 0` and `value: null`.
- `Match.personal_eliminations` is the last committed personal eliminations count, or `null` when it was never read or the Layout has no `personal_eliminations` rect. A counter missing from the Layout is not read at all, so it produces neither Events nor `unreadable`.

- `transcript` is one segment of speech (`source: "speech"`), with `text`, `language` (`en` or `pt`) and `confidence` (`exp(avg_logprob)` of the segment); `time` and `end` are its true span in seconds from the start of the Recording, so a call just after the results screen can lie beyond `Match.end`. Events of a Match are in time order, which means a segment spoken before the HUD appeared precedes `match_start`. `speaker` is the enrolled name or `unknown_N` (see Speakers below); it is `null` on every other Event. `category` (`personal` or `game_related`) and `category_confidence` (0 to 1) come from the categorization described under Categories below; both are `null` on every other Event and whenever a segment was not judged.

`recording.started_at` is an ISO 8601 time. It is read from the file name when that holds `YYYY-MM-DD_HH-MM-SS` or `YYYY-MM-DD HH-MM-SS` (OBS's names; local time, so no offset: `2026-10-09T11:59:53`), else from the container's `creation_time` tag (with its UTC offset: `2026-10-09T09:59:53+00:00`), else it is `null`. An impossible date in the name falls through to the container, then `null`.

A Recording that is cut short is an error, not a partial result: `analyse` exits 1, writes no output (and no `--observations` file) and names the file, when the header promises a duration that the decodable video falls more than 2 s short of, or when its header or first frame cannot be read.

Counter digits are about 17 px tall at 2560x1440 and Tesseract reads them reliably from 1920x1080 up; at 720p they are unreliable.

#### Layout selection

`--layout` takes one Layout file or a directory of them (`*.yaml` and `*.yml` directly inside it, no recursion; an invalid file is an error naming it). Layouts whose `aspect_ratio` differs from the Recording's by more than 2% are dropped; if none is left and the Recording shows the in-match HUD, `analyse` exits 1 with an error that names the Layouts' ratios, the Recording's `WxH` and `victorious setup` (a Recording with no HUD at all is reported as `not_fortnite` instead). Each remaining Layout's `hud_anchor` is then scored on in-match frames: one frame every `ceil(duration / 240)` s (at least 1 s; at most every 10 s in the first 10 minutes, so about 240 candidates plus at most 60) from the start is read in time order until 20 of them show an in-match HUD (some candidate Layout scores at least 0.7), and a Layout's score is its mean over those 20. The best score wins, ties go to the first file by name, and the choice does not depend on `--jobs`.

The output records `layout: {path, score}`, with `path` as given on the command line. When no Layout reaches the minimum score of 0.7, or no sampled frame scores at the anchor of any Layout although the HUD icons are on screen (a HUD no Layout describes), `analyse` exits 1, writes no output and says `no Layout matches`, naming `victorious setup`.

A Recording in which no in-match HUD appears in the first 10 minutes, for any Layout and to the HUD detector `setup` uses (the reference icons), is not Fortnite: `analyse` exits 0 with `matches: []`, `layout: null` and `warnings: [{code: "not_fortnite", message}]`, and repeats the warning on stderr. `--observations` then writes an empty file. A HUD that first appears after minute 10 is not looked for. Such a result is not cached, but only a complete Recording gets it: a truncated one is the error above. Scores are cached next to the Observations, keyed by the Recording and the set of Layout contents, so a rerun does no selection decoding; `--no-cache` disables this too. A Recording whose HUD changes design part-way averages low for every Layout and is refused.

#### Parallel decode, cache and `--observations`

`analyse` reads the Recording in whole-second chunks (60 s) on up to `--jobs` processes (default: one per CPU), each chunk decoded by its own ffmpeg. Chunk boundaries sit on whole seconds, so every chunk samples exactly the frames a single serial decode would, and the counter skip-OCR rule is replayed over the Match loop's exact schedule afterwards; the output is byte-identical for any `--jobs` and chunk size. A Recording whose chunks do not join up (missing seconds) is rejected as undecodable.

Counter digits are read in-process through `libtesseract` (the command-line `tesseract` is the fallback when the library cannot be loaded, and gives the same readings), one engine instance per concurrent reader. Whisper and the diarization embedder are sized by physical cores, not logical CPUs, when the kernel reports hyper-threading as on, and the embedder runs the parts of each batch on threads. None of this is meant to change a reading, a transcript or a Speaker: the in-process readings were checked equal to the command line's on the real counter masks, and the parts of an embedding batch are the rows of the whole up to floating-point rounding.

The stderr line ends with where the wall time went, `(layout 6.6 s, frames 27.8 s, counters 2.2 s, speech 45.8 s)`: Layout selection, decoding and scoring the sampled frames, counter OCR, and transcription with diarization. A cached run has no `frames` or `counters` phase. Measured on 2560x1440 AV1 clips of real Recordings (24 logical CPUs, cache off, speech on, a machine that was also busy with other work): 0.62 x real time before and 0.39 x after for a 10 min clip, so a 3 h Recording takes well over the 15-20 min target; the profile and what is left are in issue #30.

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

#### Speakers

Every `transcript` Event gets a `speaker`. Diarization (pyannote `speaker-diarization-community-1`, on CPU) runs once per Recording over the audio around the transcript segments only, each padded by 1 s and joined with 0.5 s of silence, with music silenced exactly as Whisper saw it (and anything outside the Match windows, which Whisper never saw, silenced too); one run over all Matches keeps a voice's identity across Matches. A segment takes the speaker whose turns overlap it most (the nearest turn when none does). `analyse --voices DIR` (default `./voices`, when it exists) enrolls known Speakers: one clip of a few seconds per person, any audio ffmpeg decodes, named by file stem (`voices/alice.wav` is `alice`). Each diarized speaker is compared with every enrolled voice by the cosine distance of WeSpeaker embeddings (the speaker's mean over its up to 10 longest turns of at least 0.5 s) and takes the nearest name within 0.6; several diarized speakers may share one name. The others are `unknown_1`, `unknown_2`, ... in order of first appearance, stable within one Recording and not across Recordings. A missing or empty `voices/` is not an error (an explicit `--voices` that does not exist adds a stderr warning, and one that is a file exits 1); a clip that cannot be decoded, a stem starting `unknown_`, or two clips with one stem exits 1 before the Recording is read. `speaker` is `null` only when diarization finds no speech at all. The stderr line reports `speakers N (enrolled M)`. The pipeline (about 35 MB, pinned to a revision of the ungated `pyannote-community` mirror, see `docs/adr/0004-ungated-diarization-mirror.md`) is downloaded on first use and needs no token.

#### Categories

Each `transcript` Event is judged personal or game-related by TypeSafe's Jev, asked as one Choice question (`personal` or `game_related`) about the segment text alone; `category` is the option chosen and `category_confidence` is the confidence Jev returns with that answer (distinct from the Whisper `confidence` of the Event). The API key is read from `TYPESAFE_API_KEY`; there is no command-line option for it. `VICTORIOUS_JEV_URL` overrides the endpoint (default `https://api.typesafe.ai/v1/systemone`, model `jev-latest`); it must be `https`, or `http` to `localhost`/`127.0.0.1`/`::1`, and redirects are not followed, so the key is only ever sent to that URL.

Only the text of each segment is sent: no audio, no times, no speaker, no neighbouring segments. The text of casual personal conversation therefore leaves the machine (see `docs/adr/0002-external-categorization-of-transcript-text.md`); transcription and diarization stay local. Not setting the key, or passing `--no-speech`, sends nothing.

Without a key, or when Jev fails, the category is `null` and the run still succeeds with exit 0; there is no heuristic fallback. A key that is unset or blank prints a one-line `note` on stderr when there are segments. Each distinct text is asked once per run, up to 8 requests in parallel with a 30 s timeout; 429, 529, 5xx and network errors are retried (3 attempts in all, 0.5 s then 1 s apart, or the `Retry-After` of the reply when longer, up to 10 s), other 4xx responses and unusable replies leave that one segment `null`. A rejected key (401 or 403), or five failed segments in a row (whatever the failure), stops asking about the rest of the run, which stay `null`, and a `warning` on stderr names the first failure. The stderr line reports `categories K of M`.

Speech needs a Hugging Face token (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or `hf auth login`); without one `analyse` exits 1 before reading the Recording and says so. The token is only checked for presence here. The audio tagger, the diarization pipeline and the `--voices` clips are loaded (the models downloaded on first use) at the same point, so a missing or corrupt model or clip also fails before the Recording is read. The Whisper model is public: it is downloaded on first use into the Hugging Face cache (about 480 MB) and loaded from it afterwards. A Recording without audio, or an `--audio-stream` it does not have, is an error that names `--no-speech`.

`--no-speech` skips all of this (setting `VICTORIOUS_NO_SPEECH` to `1`, `true` or `yes` makes it the default): no token is needed and no speech library is imported. Transcripts are not part of the Observation cache; a cached rerun still transcribes.

### Review

```bash
uv run victorious review analysis.json                       # serve the page, save analysis.labels.json
uv run victorious review analysis.json --against labels.json # compare, exit 1 when they differ
```

`review` reads the JSON `analyse` wrote and serves a page on `http://127.0.0.1:PORT/TOKEN/` (printed on stderr, opened in your browser unless `--no-browser`). Each Match is a card with its start and end, whether it was a Win, the placement and the counters read, and two frames of the Recording: the one at its start and the results screen two seconds after the first results second (the banner is still animating at that second). A Match that never reached a results screen says so and shows its last in-match frame. Confirm a Match when what was read is right and reject it when it is not a real Match or a value is wrong; `c` and `r` decide the focused card. `Save labels` is enabled once every Match has a verdict. It writes the labels file, then the server exits 0; `Ctrl-C` cancels (exit 1, nothing written).

| Option | Meaning |
|--------|---------|
| `--recording PATH` | the Recording, when it is not at the path written in the JSON (relative to the working directory) |
| `-o`, `--output PATH` | where the labels go (default: `<analysis stem>.labels.json` beside the JSON); an existing file is replaced |
| `--no-browser`, `--port N` | only print the address; listen on this port instead of a free one |
| `--against LABELS` | compare the analysis with a labels file instead of serving a page |
| `--tolerance SECONDS` | with `--against`, how far a Match boundary may move (default 2, since analysis samples at one frame per second) |

The labels file is JSON with its own `labels_version` (`1.0`), separate from the analysis schema: the Recording (`path`, `duration`), the analysis it came from, and one entry per Match with `start`, `end`, `verdict` (`confirmed` or `rejected`) and the `win`, `placement` and `personal_eliminations` that were shown. It is meant to be kept beside a Recording as a regression fixture.

`--against` pairs each Match with a label by time, not by index: spans that overlap by at least half are the same Match. It prints one line per difference, `kind  start-end  detail`, then a summary:

| Kind | Meaning |
|------|---------|
| `missing` | a confirmed Match is no longer found |
| `changed` | `win`, `placement` or `personal_eliminations` differ from a confirmed label; or a rejected Match is now read differently and needs a new review |
| `shifted` | start or end moved by more than `--tolerance` |
| `reappeared` | a rejected Match is found again with the same values |
| `unlabelled` | a Match with no label, for example a new detection |

It prints `no differences (N matches)` and exits 0 when clean, and exits 1 when there are differences. Labels for another Recording (a different file name, or a duration more than 1 s apart), a corrupt labels file or an analysis of another schema version exit 1 with a message on stderr. `--against` with `-o`, `--recording`, `--no-browser` or `--port` is a usage error (exit 2). Known limits: only detected Matches can be labelled (a Match the analysis missed cannot be added), and a Match split in two by a HUD dropout shows as `missing` plus `unlabelled`.

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
