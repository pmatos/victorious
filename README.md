# Victorious

Turns long Fortnite gameplay Recordings into a timestamped record of what happened in each Match: Wins, final placement, eliminations and speech.

All five subcommands work. The remaining work is tracked in [epic #13](https://github.com/pmatos/victorious/issues/13).

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

`setup` knows three HUD designs and records the one it found as `hud_design` in the Layout: the counters in an icon column at the top right (`icon_column`), the same column with only two rows, personal eliminations and players left, as some game modes draw it (`icon_pair`: no team eliminations, so the Layout has no `team_eliminations` rectangle), or a horizontal row of icons under the minimap (`counter_row`: players left, personal eliminations, team eliminations). All are found without telling it which. Give `setup` seconds from the match itself: the countdown before the bus launches shows more rows than the match does.

Keep the Layouts in one folder (for example `layouts/`): `analyse` picks the one that fits each Recording, which [many Recordings](#6-analyse-many-recordings) rely on. Set up again only for a new HUD design or aspect ratio.

### 2. Analyse

```bash
uv run victorious analyse recording.mkv --layout layout.yaml --no-speech -o analysis.json
```

Start with `--no-speech`: it needs no token and no models, and gives the Matches, Wins, placement and counters. `--layout` also takes a directory of Layouts (one per HUD design or aspect ratio); the best fit for the Recording is chosen. Without `-o` the JSON goes to stdout. `--observations observations.jsonl` also writes every raw HUD reading, one line per element per second, which is what to read when a counter looks wrong.

Reruns on the same Recording and Layout reuse the cached Observations (`--no-cache` to ignore them, `--jobs N` to limit the processes). For a folder of Recordings, see [step 6](#6-analyse-many-recordings); for how long a run takes, [how long a run takes](#how-long-a-run-takes).

### 3. Read the output

Every time is in seconds from the start of the Recording. The output below is from a 10 minute cut that starts in the middle of a Match (so the first Match starts at `0.0`); the second Match is one the player left before its results screen.

```bash
jq -c '.matches[] | {start, end, win, placement, personal_eliminations}' analysis.json
```

```text
{"start":0.0,"end":144.0,"win":false,"placement":2,"personal_eliminations":4}
{"start":218.0,"end":265.0,"win":false,"placement":null,"personal_eliminations":0}
{"start":355.0,"end":599.0,"win":false,"placement":null,"personal_eliminations":0}
```

Each Match also has an `events` list in time order: `match_start`, `personal_eliminations` and `team_eliminations` (with the new count in `value`), `players_left`, `win`, `final_placement` (`value` is the placement), `eliminated` (the player went out, see [Reading the Events](#reading-the-events)), `unreadable`, `match_end` and, with speech, `transcript`. `placement: null` means the Match was left before a results screen, was still running at the end of the Recording, or its placement could not be read. A `warnings` entry of `not_fortnite` (see [Common errors](#common-errors)) means no in-match HUD was found, so `matches` is empty.

```bash
uv run victorious schema -o analysis.schema.json
```

`schema` prints the JSON schema of this output (to stdout without `-o`), for validating or generating types.

### 4. Add speech (transcripts, Speakers, categories)

Speech is on unless `--no-speech` is given, and needs this set up once:

1. **Hugging Face token.** Create a read-only token at <https://huggingface.co/settings/tokens> and either `export HF_TOKEN=hf_...` (or `HUGGING_FACE_HUB_TOKEN`) or run `hf auth login`. Every model used is public, so no access request is needed; the tool only checks that a token exists, and without one `analyse` exits 1 before reading the Recording. Never paste the token into a report (see [Reporting feedback](#reporting-feedback)).
2. **First-run downloads.** The first speech run downloads, into the Hugging Face cache (`~/.cache/huggingface`), the audio tagger that finds music (YAMNet, about 16 MB), the diarization pipeline (the `pyannote-community` mirror, about 35 MB) and, when the first Match window is transcribed, Whisper `large-v3-turbo` (about 1.6 GB, with no progress line of ours, so a long silent pause there is the download). Later runs load them from the cache; a machine that is offline on its first run fails with a message naming `--no-speech`.
3. **Enrolled voices (optional).** Put one clip of a few seconds per person in a `voices/` folder in the directory you run the tool from, named by the person: `voices/alice.wav` becomes the Speaker `alice`. Any other voice is `unknown_1`, `unknown_2`, ... (numbered by first appearance, so stable within one Recording and not across Recordings). A missing `voices/` is not an error; use `--voices DIR` for another folder, and run every Recording of a [batch](#6-analyse-many-recordings) from the same directory so they all see the same voices.
4. **Jev API key.** `export TYPESAFE_API_KEY=...` makes each transcript `category` `personal` or `game_related`. Only the text of each segment is sent to the external service (see [ADR 0002](docs/adr/0002-external-categorization-of-transcript-text.md)). Without a key `analyse` exits 1 before reading the Recording; pass `--no-categories` (or set `VICTORIOUS_NO_CATEGORIES=1`) to run speech anyway with every `category` and `category_confidence` `null`. Segments Jev cannot judge are `null` too, and the run still exits 0.

```bash
export HF_TOKEN=hf_...
uv run victorious analyse recording.mkv --layout layouts/ -o analysis.json
```

The first audio stream is transcribed; if the voices (microphone, party chat) are on another track, pass `--audio-stream N` (counted from 0 among the audio streams). Whisper and diarization run on an NVIDIA GPU when one is usable (stderr then says `victorious: speech: Whisper on the GPU (CUDA)`), else on the CPU; `VICTORIOUS_DEVICE=cpu|cuda` forces one (see [Speech on an NVIDIA GPU](#speech-on-an-nvidia-gpu)). Speech takes most of the run time (see [how long a run takes](#how-long-a-run-takes)), so keep `--no-speech` (or `VICTORIOUS_NO_SPEECH=1`) while iterating on a Layout. Transcripts are not cached: a rerun transcribes again even when the frames come from the cache.

### 5. Review the Matches

```bash
uv run victorious review analysis.json
```

A page opens (the address is printed on stderr; `--no-browser` only prints it) with one card per Match and two frames of the Recording. Confirm a Match when what was read is right, reject it when it is not a real Match or a value is wrong (`c` and `r` decide the focused card). `Save labels` writes `analysis.labels.json` beside the JSON and the command exits 0; `Ctrl-C` cancels and writes nothing. If the Recording moved, pass `--recording PATH`.

Keep the labels file beside the Recording. After changing the tool or the Layout, rerun `analyse` and compare:

```bash
uv run victorious review analysis.json --against analysis.labels.json
```

It prints one line per difference (`missing`, `changed`, `shifted`, `reappeared`, `unlabelled`) and exits 1, or prints `no differences (N matches)` and exits 0.

```text
changed     0:00:00-0:02:24  placement: expected 2, found 3
missing     0:05:55-0:09:59  confirmed, and not found
2 differences (2 matches)
```

`--tolerance SECONDS` sets how far a boundary may move (default 2).

### 6. Analyse many Recordings

To test the tool on a folder of Recordings, run one `analyse` per file with its own `-o`, so every Recording has its own JSON and its own stderr log. Run them one after another: each `analyse` already uses every CPU (`--jobs`), and the GPU has only a few decode engines.

```bash
mkdir -p results
for recording in recordings/*.mkv; do
  name=$(basename "${recording%.*}")
  rm -f "results/$name.analysis.json"
  if uv run victorious analyse "$recording" --layout layouts/ --no-speech \
      -o "results/$name.analysis.json" 2> "results/$name.log"; then
    echo "ok      $name"
  else
    echo "FAILED  $name (exit $?)"
  fi
done
```

`mkdir -p` comes first because `analyse` refuses an output folder that does not exist, and `rm -f` clears an older result so that a failure leaves no stale JSON behind. Quote `"$recording"`: OBS names can hold a space (`2026-10-09 11-59-53.mkv`). Add `recordings/*.mp4` to the list for other containers. Drop `--no-speech` (and `cd` to the folder that holds `voices/`) for a speech pass once the frames look right; see [how long a run takes](#how-long-a-run-takes) before running that on many hours of video.

- **One Layout per HUD design and aspect ratio.** `--layout layouts/` takes the whole folder, and `analyse` picks, for each Recording, the Layout whose aspect ratio fits and whose HUD scores best (the choice is in `layout` of the JSON and on stderr). A Layout serves every Recording with the same HUD design and aspect ratio, at any resolution.
- **Another HUD design or aspect ratio.** The Recording fails with exit 1 and an error that names `victorious setup` when it shows an in-match HUD (one that shows none is a `not_fortnite` warning with exit 0 instead): `no Layout fits this 1920x1080 Recording (...)` when no Layout has its aspect ratio, `no Layout matches this Recording's HUD: ...` when the aspect ratio fits but the HUD differs. Run `setup` on that Recording (see step 1), save the new Layout in the same `layouts/` folder and run the loop again; a Recording that already worked keeps its Layout unless the new one scores better on it.
- **Spotting failures.** The exit code is the verdict: `0` is a result, `1` is an error (the reason is the last line of `results/NAME.log`, `victorious analyse: <reason>`) and nothing is written for that Recording, so a missing `results/NAME.analysis.json` is a failure too. A run that exits `0` can still deserve a look: `warnings` in the JSON (today only `not_fortnite`: no in-match HUD in the first 10 minutes, so `matches` is empty and `layout` is `null`, repeated on stderr as `victorious analyse: warning: not_fortnite: ...`) and Matches with `placement: null` (see below).

One line per Recording summarises the whole folder:

```bash
jq -r '[input_filename, (.matches | length), ([.matches[] | select(.win)] | length),
        ([.matches[] | select(.placement == null)] | length),
        (.warnings | map(.code) | join(","))] | @tsv' results/*.analysis.json
```

```text
results/notgame.analysis.json	0	0	0	not_fortnite
results/out.analysis.json	3	0	2	
results/win.analysis.json	1	1	0	
```

The columns are the file, the Matches, the Wins, the Matches with no placement and the warnings. A Match with no placement ended without a results screen (the player left it) or was still running at the end of the Recording, or its placement could not be read: open `review` on that one. A Recording with no Matches and no warning is worth a look too.

To list the Recordings that have no result yet (to retry the failures after adding a Layout; use the same file list as the loop above, `.mp4` included):

```bash
for recording in recordings/*.mkv; do
  [ -e "results/$(basename "${recording%.*}").analysis.json" ] || echo "$recording"
done
```

and the reason for each failure is the last line of its log: `tail -n 1 results/NAME.log`.

### 7. Cut the Wins into one video

```bash
uv run victorious extract analysis.json --wins -o wins.mkv
```

`extract` reads the JSON `analyse` wrote, cuts every Win out of the Recording and joins them, in order, into `wins.mkv`. Each cut runs from the start of the Match to a few seconds after its results screen appears, so the Victory Royale screen is in the video. The streams are copied, not re-encoded, so it takes seconds (two Wins of 19 minutes each from a 2560x1440 Recording took 20 s) and the video keeps the Recording's quality, but each cut starts at the keyframe at or before its Match start: a few seconds of lobby or bus before the Match. Add `--exact` to re-encode so each cut starts on its exact second; that took about 35 minutes for the same two Wins. Reading the output is covered under [Extract](#extract).

### How long a run takes

Times below were measured on three real 2560x1440 AV1 Recordings of 61 to 83 minutes, on a machine with 24 logical CPUs and VAAPI hardware decode. Treat them as indicative: the machine was also busy with other work, and times on an overloaded one stretch a lot (frames-only runs of 10 minute cuts took 197 s and 233 s while its load average was about 40; an 8 minute cut took 80 s once the load fell).

| Run | Time |
|-----|------|
| frames only (`--no-speech`), cold cache | about 0.11 x real time (0.106 x to 0.116 x on the three Recordings); a 3 h Recording is about 20 min |
| speech on, measured before ADR 0006 (Whisper `large-v3-turbo`, batched decoding), one 61 min Recording, loaded machine | 2295 s (38 min), about 0.63 x real time |
| the same run by phase (measured before ADR 0006) | layout 22 s, frames 320 s, counters 21 s, speech 1895 s; 34 of 61 chunks decoded on the GPU; 592 transcript segments (the sequential decoder gives more, shorter segments, so this count is stale) |
| speech on with sequential decoding (current), same Recording | not measured on a whole Recording; projection from [ADR 0006](docs/adr/0006-sequential-whisper-decoding.md): speech 1895 s x 1.4 = about 2650 s, so about 3050 s (51 min), about 0.83 x real time; the 1.4 is an average (1.0 to 2.0 per pair, 1.8 on a busier machine), so expect more on a loaded one |
| frames-only rerun of the same Recording and Layout | from the cache, no `frames` or `counters` phase: 0.1 s on an 8 minute cut |
| speech-on rerun | still pays the whole speech phase: transcripts are not cached |
| speech on an NVIDIA GPU (RTX 4090, `--no-categories --no-cache`, 2560x1440 h264, another machine) | 19.6 min Recording: 107 s total, speech 44 s (522 s and 459 s on its CPU); 28.9 min Recording: 136 s, speech 45 s (665 s and 576 s on its CPU) |

Speech is about 83% of the measured speech-on run (about 87% with the sequential decoding, projected), and it grows with the time inside Match windows (each Match plus 10 s either side), not with the length of the Recording. `large-v3-turbo` is the default for accuracy ([ADR 0005](docs/adr/0005-large-v3-turbo-default-whisper-model.md)), decoded in order with the previous text as context ([ADR 0006](docs/adr/0006-sequential-whisper-decoding.md)), which costs about 1.4 x the time of the batched decoding the two measured speech rows above used (a ratio from paired runs on short windows, not from a whole Recording). At the ratio above, with that factor on the speech share, a 3 h Recording with speech would take close to 2.5 h (a projection from one Recording, not a measurement: a Recording with less time inside Matches takes less), so do the frames-only pass first, check it with `review`, and add speech after. The cache keeps its Observations in `~/.cache/victorious` and is keyed by the Recording and the Layout, so running again with nothing changed costs seconds; `--no-cache` ignores it. The stderr summary line ends with the phases; these are a cold run and its rerun on an 8 minute cut (speech off), where the fixed Layout selection weighs more than on a long Recording:

```text
victorious analyse: layout layout.yaml (score 0.99, 1 candidate), cache miss, 79.9 s (layout 12.4 s, frames 63.5 s, counters 2.7 s)
victorious analyse: decode: 9 of 9 chunks on the GPU, 0 on the CPU
victorious analyse: layout layout.yaml (score 0.99, 1 candidate), cache hit, 0.1 s (layout 0.0 s)
```

### Hardware decode

A GPU is used for decoding when one works, on Windows and Linux, and nothing needs to be set up: `analyse` tries an NVIDIA GPU through CUDA (when the NVIDIA driver is installed) and then VAAPI on a `/dev/dri/renderD*` device (AMD or Intel on Linux), keeps one only when two probe frames come out byte for byte the same as software decoding (so results never depend on it) and otherwise decodes in software without saying anything. It needs an 8-bit 4:2:0 Recording, and for NVIDIA an ffmpeg built with CUDA (the Windows builds from gyan.dev and BtbN are).

- A run that used the GPU prints one more stderr line saying how many chunks (60 s of video each) went where, for example `victorious analyse: decode: 34 of 61 chunks on the GPU, 27 on the CPU`. No such line means software decode only. A failed GPU decode costs one chunk, which is finished in software (`..., 2 resumed on the CPU after a GPU failure`), and prints one `victorious: hardware decode ...; using software decode` line; the exit code and the output are the same.
- `VICTORIOUS_HWACCEL=off` forces software decode (use it when comparing runs or when the GPU misbehaves); `VICTORIOUS_HWACCEL=cuda` or `vaapi` insists on that kind of GPU and says so when it cannot be used.
- `VICTORIOUS_HW_SLOTS=N` sets how many chunks decode on the GPU at once (default 8 on an NVIDIA GPU, 2 on VAAPI; `0` is software); a chunk that finds no free slot decodes on the CPU. `VICTORIOUS_CUDA_DEVICE=1` picks another NVIDIA GPU (by its `nvidia-smi` index), `VICTORIOUS_VAAPI_DEVICE=/dev/dri/renderD129` another VAAPI device.
- If the note says a VAAPI device cannot be created, see the troubleshooting paragraph under [Parallel decode](#parallel-decode-cache-and---observations).

### Reading the Events

The `events` of each Match are in time order; `time` is in seconds from the start of the Recording.

| `kind` | Meaning |
|--------|---------|
| `match_start`, `match_end` | the Match opens on the in-match HUD and closes on its results screen (or at its last in-match second when it was left) |
| `win` | the Victory Royale screen; `final_placement` carries the placement in `value` |
| `eliminated` | the player went out (see below) |
| `personal_eliminations`, `team_eliminations` | a kills counter changed; `value` is the new count |
| `players_left` | the players-left counter changed; `value` is the new count |
| `unreadable` | a counter (`element`) had no usable reading for more than 10 s, from `time` to `end`; `value` is `null` |
| `transcript` | one segment of speech with `text`, `language`, `speaker` and `category`; only with speech |

`eliminated` is the player themselves going out, which is not the same as `personal_eliminations` (the kills the player made). It is emitted when the white SPECTATING label shows (the squad plays on and the player watches a teammate), with `value: null`, and also on the "YOU PLACED #N" screen when the player was not already out, with the placement in `value`. A teammate's reboot card brings the player back, so one Match can have several `eliminated` Events, one per time the player goes out. Expect the last of them just before `final_placement` and `match_end` in a Match the squad lost. `unreadable` says a counter had no usable reading for a while (hidden, covered or too unclear to read); the Matches and the other counters around it are unaffected.

```bash
jq -c '.matches[] | {start, end, events: [.events[] | select(.kind == "eliminated" or .kind == "unreadable") | {kind, time, end, value}]}' analysis.json
```

```text
{"start":0.0,"end":144.0,"events":[{"kind":"eliminated","time":144.0,"end":null,"value":2}]}
{"start":218.0,"end":265.0,"events":[]}
{"start":355.0,"end":599.0,"events":[]}
```

### Reporting feedback

Open an issue at <https://github.com/pmatos/victorious/issues> (or send the files to the owner) and say what you expected and what you got. Most problems can be reproduced from these, so attach what applies:

- **The Analysis JSON** (`results/NAME.analysis.json`) and the **stderr summary line** (the `victorious analyse: layout ... cache ... (layout ... s, frames ... s, ...)` line and the `decode:` line, or the whole `results/NAME.log`).
- **The labels file** from `review` (`NAME.labels.json`) when the Matches are wrong: it says which Matches you confirmed and which you rejected. Say which Match, with its start and end, and what it should be.
- **A `--observations` dump** (`analyse ... --observations NAME.observations.jsonl`) when a counter, placement or Win looks wrong: it is every raw reading of every second, so it shows what was read where. It is large (tens of thousands of lines for a long Recording): `gzip` it.
- **The Recording's resolution and codec**, which `ffprobe` prints:

  ```bash
  ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate -show_entries format=duration -of default=nw=1 recording.mkv
  ```

- **For a HUD problem** (a Layout that does not fit, a `no Layout matches` error, wrong counters), the **`setup --preview` PNG** (`layout.preview.png`, or the path you gave with `--preview`) and the Layout file, and the `--at` times you used.
- The exact command line, and the output of `git rev-parse --short HEAD` (the version of the tool).

Do not attach your Hugging Face token or `TYPESAFE_API_KEY`. A speech-on Analysis JSON contains the text of what was said, including personal conversation; read it before posting it in a public issue, or share it privately, or send the frames-only JSON. Every Analysis JSON also holds the full path of the Recording (`recording.path`), which can contain your user name; edit it before posting.

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
| `setup: placement: no gold #N (best match 0.57)` (exit 1) | The gold `#N` beside the Victory Royale banner was not matched well enough (0.7 is needed). Seen on the orange banner of both Wins of a real 2560x1440 Recording, at every second of the results screen. Try another `--at results=` second, a Win with the other banner colour, or place `placement` with `--manual` (a rectangle that does not read back is still saved, with a warning). |
| `victorious: hardware decode ...; using software decode` | The GPU failed and the rest of that chunk was decoded in software; the result is the same. `VICTORIOUS_HWACCEL=off` skips the GPU, see [Hardware decode](#hardware-decode). |
| `no TYPESAFE_API_KEY set ...` (exit 1) | Speech is on but there is no Jev key. Set `TYPESAFE_API_KEY`, or pass `--no-categories` (every `category` is then `null`), see step 4. |
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
| `extract` | Cut the Wins of an analysis out of its Recording and join them into one video | working (`--wins`) |
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

Either kind may be repeated, and the best match of each element wins. A results frame shows only one style: a Win has the banner and the gold `#N` (`results_banner`, `placement`), an eliminated player sees a white `YOU PLACED #N` (`elimination_placement`). Pass one of each to get all three; without an eliminated frame the Layout is still written and a warning on stderr names `elimination_placement`. A counter that cannot be read on a frame is reported by name: try another `--at hud=` second where the digits are crisp. The seconds either side of each `--at` are also tried, so one unlucky video frame does not fail a counter. When nothing is found on a results second, or an element is found at a size `analyse` does not read (the `YOU PLACED #N` text animates in size), `setup` keeps reading further out, up to 8 s either side, and prefers a detection at a readable size over a better-scoring one.

When a required element cannot be found, `setup` exits 1, names each one with the reason, points to `--manual` and writes no Layout (the preview is still written when anything was found). A missing `--at` kind, or a time that is not a time, is a usage error (exit 2). `setup` finds HUDs between 0.7x and 1.4x of the reference size, but `analyse` reads only 0.9x to 1.1x, so a Layout that would not read back is reported instead of written. `analyse` reads the Victory Royale banner between 0.9x and 1.6x, which covers its large, lower position when it appears, but `setup` finds it only up to 1.4x and `analyse` cannot read it below 0.9x. The banner animates: pick a second just after it appears, while it is full size, since on some XP screens it shrinks below 0.9x and `setup` then reports it.

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

A Match starts when the in-match HUD appears and ends at its results screen; being eliminated while the squad plays on, or losing the HUD while spectating, does not end it (the player gets an `eliminated` Event, see below, and the Match ends at the squad's results screen). The white "YOU PLACED #N" screen is the results screen, so it ends the Match: on the Recordings it was checked on it appears only once the whole squad is out, straight after the in-match HUD. A Match left before its results screen (the player quit in the lobby, on the bus or mid-game) ends at its last in-match second (the last second of the SPECTATING label when it left while spectating), with `win: false` and `placement: null`, once the HUD returns after more than 20 seconds with the players-left counter climbing the way it does as a new Match's lobby fills; a HUD run of under 30 seconds that only showed a lobby filling (the player left before the game) is not a Match and is not reported; the HUD returning with the counters where they were is the same Match. A Match still running when the Recording ends is reported with `win: false` and `placement: null`. A results screen whose placement cannot be read also gives `placement: null`.

The output carries `schema_version` (1.7), `recording` (`path`, `duration` and `started_at`, see below), `layout` (the Layout used and its score, see below), `warnings` and, per Match, `index`, `start`, `end`, `win`, `placement`, `personal_eliminations` and its Events. Every Event has `kind`, `time`, `source`, `confidence`, `value`, `end`, `element`, `text`, `language`, `speaker`, `category` and `category_confidence`.

- `match_start`, `win`, `final_placement` (`value` is the placement) and `match_end` come from the Match boundaries and the results screen.
- `eliminated` is the player going out, as distinct from `personal_eliminations`, which counts the player's kills. While teammates play on, the player's screen shows the white "SPECTATING" label (above the name of the teammate watched) instead of the HUD; `eliminated` is emitted at the first of two consecutive seconds that show it, with `value: null` since no placement is on screen then, even if the squad is out later (the fade-in of the label can make it one or two seconds later than the death). The player is out until the HUD has been back for 2 seconds, so a map or Social overlay or a one second HUD flicker in between is the same elimination, and a teammate's reboot card that brings the player back makes the next elimination a new Event: a Match can have several. A "YOU PLACED #N" screen reached by a player who was not already out is an `eliminated` too, at the results second and with the placement in `value`, just before `final_placement` and `match_end`. The label is searched in a fixed band at the top centre of the frame, so no Layout field is involved and `setup` does not need to be rerun, and it only counts within 20 s of the HUD last being seen. A Match left while spectating, with no results screen, ends at the last second the label showed instead of at the last HUD second. No `eliminated` is emitted for a Win the player was not out for.
- `personal_eliminations`, `team_eliminations` and `players_left` are HUD counters, read with Tesseract once per sampled second. An Event is emitted only when the value changes, with the new count in `value` and the first second it was read. A value is committed only after it is read in two consecutive readable seconds, so one misread frame never becomes an Event, and the first value read in a Match is a silent baseline, not an Event. A change of any counter that goes back to the value it left within 5 s (a 2 s misread of `6` as `5` is `6 -> 5 -> 6`, of `2` as `9` is `2 -> 9 -> 2`) is a misread too: neither the change nor the return becomes an Event, in either direction, and the Events around it are unaffected. A fall of `personal_eliminations` or `team_eliminations` to a value above 0 that is read at an average confidence below 0.5 is a misread as well and is not an Event; a confident fall is kept, and so is a fall to 0 (the reset of a new Match) whatever its confidence, because Tesseract reads a genuine 0 at low confidence on some builds. The count that does not return within 5 s is a real change and stays, so a fall that holds is one Event at the first second it was read, and a rise while the lobby fills is kept. When a counter moves by more than one between two sampled seconds, only that gap is decoded again at 4 fps and every step gets its own Event, timed to the first readable 0.25 s slot in which it appears (an unreadable or misread sample inside the window can delay it by a slot); steps that fall in the same slot share its time, and the number of Events equals the number of steps. This applies to rises of up to 3 in `personal_eliminations` and `team_eliminations` and falls of up to 10 in `players_left`; larger or wrong-direction jumps are OCR noise candidates and stay one Event. A gap is not re-sampled when a second inside it was unreadable, when it spans more than 3 s, or after an `unreadable` span: those jumps stay one Event at the first second the new value was read. The eliminations counters never fall within a Match, so a rise only gets steps beyond the highest value already committed; a recovery to that value after a sustained misread stays one Event. Single-step changes cause no extra decoding. Values are never interpolated.
- `unreadable` marks a counter (`element`) with no usable reading for more than 10 s, from `time` to `end` (a span of exactly 10 s is not reported). It has `confidence: 0` and `value: null`.
- `Match.personal_eliminations` is the last committed personal eliminations count, or `null` when it was never read or the Layout has no `personal_eliminations` rect. A counter missing from the Layout is not read at all, so it produces neither Events nor `unreadable`.

- `transcript` is one segment of speech (`source: "speech"`), with `text`, `language` (`en` or `pt`) and `confidence` (`exp(avg_logprob)` of the segment); `time` and `end` are its true span in seconds from the start of the Recording, so a call just after the results screen can lie beyond `Match.end`. Events of a Match are in time order, which means a segment spoken before the HUD appeared precedes `match_start`. `speaker` is the enrolled name or `unknown_N` (see Speakers below); it is `null` on every other Event. `category` (`personal` or `game_related`) and `category_confidence` (0 to 1) come from the categorization described under Categories below; both are `null` on every other Event and whenever a segment was not judged.

`recording.started_at` is an ISO 8601 time. It is read from the file name when that holds `YYYY-MM-DD_HH-MM-SS` or `YYYY-MM-DD HH-MM-SS` (OBS's names; local time, so no offset: `2026-10-09T11:59:53`), else from the container's `creation_time` tag (with its UTC offset: `2026-10-09T09:59:53+00:00`), else it is `null`. An impossible date in the name falls through to the container, then `null`.

A Recording that is cut short is an error, not a partial result: `analyse` exits 1, writes no output (and no `--observations` file) and names the file, when the header promises a duration that the decodable video falls more than 2 s short of, or when its header or first frame cannot be read.

Counter digits are about 17 px tall at 2560x1440 and Tesseract reads them reliably from 1920x1080 up; at 720p they are unreliable.

#### Layout selection

`--layout` takes one Layout file or a directory of them (`*.yaml` and `*.yml` directly inside it, no recursion; an invalid file is an error naming it). Layouts whose `aspect_ratio` differs from the Recording's by more than 2% are dropped; if none is left and the Recording shows the in-match HUD, `analyse` exits 1 with an error that names the Layouts' ratios, the Recording's `WxH` and `victorious setup` (a Recording with no HUD at all is reported as `not_fortnite` instead). Each remaining Layout's `hud_anchor` is then scored (with the icons of that Layout's `hud_design`, so a folder with an icon-column Layout and a counter-row Layout picks the one the Recording uses) on in-match frames: one frame every `ceil(duration / 240)` s (at least 1 s; at most every 10 s in the first 10 minutes, so about 240 candidates plus at most 60) from the start is read in time order until 20 of them show an in-match HUD (some candidate Layout scores at least 0.7), and a Layout's score is its mean over those 20. The best score wins, ties go to the first file by name, and the choice does not depend on `--jobs`.

The output records `layout: {path, score}`, with `path` as given on the command line. When no Layout reaches the minimum score of 0.7, or no sampled frame scores at the anchor of any Layout although the HUD icons are on screen (a HUD no Layout describes), `analyse` exits 1, writes no output and says `no Layout matches`, naming `victorious setup`.

A Recording in which no in-match HUD appears in the first 10 minutes, for any Layout and to the HUD detector `setup` uses (the reference icons), is not Fortnite: `analyse` exits 0 with `matches: []`, `layout: null` and `warnings: [{code: "not_fortnite", message}]`, and repeats the warning on stderr. `--observations` then writes an empty file. A HUD that first appears after minute 10 is not looked for. Such a result is not cached, but only a complete Recording gets it: a truncated one is the error above. Scores are cached next to the Observations, keyed by the Recording and the set of Layout contents, so a rerun does no selection decoding; `--no-cache` disables this too. A Recording whose HUD changes design part-way averages low for every Layout and is refused.

#### Parallel decode, cache and `--observations`

`analyse` reads the Recording in whole-second chunks (60 s) on up to `--jobs` processes (default: one per CPU), each chunk decoded by its own ffmpeg. Chunk boundaries sit on whole seconds, so every chunk samples exactly the frames a single serial decode would, and the counter skip-OCR rule is replayed over the Match loop's exact schedule afterwards; the output is byte-identical for any `--jobs` and chunk size. A Recording whose chunks do not join up (missing seconds) is rejected as undecodable.

Counter digits are read in-process through `libtesseract` (the command-line `tesseract` is the fallback when the library cannot be loaded, and gives the same readings), one engine instance per concurrent reader. Whisper and the diarization embedder are sized by physical cores, not logical CPUs, when the kernel reports hyper-threading as on, and the embedder runs the parts of each batch on threads. None of this is meant to change a reading, a transcript or a Speaker: the in-process readings were checked equal to the command line's on the real counter masks, and the parts of an embedding batch are the rows of the whole up to floating-point rounding.

Frames are decoded on the GPU when one works, and in software otherwise; nothing needs to be set up. An NVIDIA GPU is tried first, through CUDA (NVDEC), when `nvidia-smi` is on `PATH` or a render device is NVIDIA's (`VICTORIOUS_CUDA_DEVICE` picks a GPU by its `nvidia-smi` index and asks for it), then VAAPI. On first use `analyse` decodes two frames of the Recording both ways and keeps the GPU only when they are byte for byte the same, so the output (Observations, Events, cache entries) never depends on it. It needs an 8-bit 4:2:0 Recording. VAAPI needs a `/dev/dri/renderD*` device (`VICTORIOUS_VAAPI_DEVICE` picks one) and `analyse` sets `LIBVA_DRIVER_NAME` for the ffmpeg subprocess only: your own value is tried first, then `radeonsi` on AMD or `iHD` and `i965` on Intel. The GPU has few decode engines, so at most 8 chunks decode on an NVIDIA GPU at once and 2 on a VAAPI device (`VICTORIOUS_HW_SLOTS` changes that, 0 means software); a chunk that finds every slot busy decodes in software, except the last ones, which wait. On an RTX 4090 the frames of a 3 h Recording took 462 s with 4 slots, 434 s with 8 and 893 s with every chunk on the GPU: its decoder is then the limit, and the CPU decoding beside it adds speed. That cuts the CPU time of decoding a 2560x1440 AV1 Recording about ten-fold (the GPU decodes about 8x real time there) and leaves the cores to OCR. If the GPU fails, even part-way through a chunk, `analyse` prints one `victorious: hardware decode ...; using software decode` line on stderr, finishes that chunk in software with the same exit code and tries the GPU again on the next one; after three failures in a row a process gives that GPU up and tries the next one (VAAPI after an NVIDIA GPU), else decodes in software. A run that used the GPU ends with a stderr line saying how many chunks it decoded (`victorious analyse: decode: 34 of 61 chunks on the GPU, 27 on the CPU`, with `, N resumed on the CPU after a GPU failure` added when a chunk fell back part-way). `VICTORIOUS_HWACCEL=off` (or the hidden `--hwaccel off`) forces software; `cuda` or `vaapi` insists on that GPU and says so when it cannot be used. Troubleshooting (VAAPI): if the note says the device cannot be created, run `LIBVA_DRIVER_NAME=radeonsi ffmpeg -init_hw_device vaapi=va:/dev/dri/renderD128 -f lavfi -i nullsrc -frames:v 1 -f null -` (or `iHD`) to find the driver that works, and export it; a `LIBVA_DRIVER_NAME` left over from another GPU is the usual reason ffmpeg finds no device.

The stderr line ends with where the wall time went, `(layout 6.6 s, frames 27.8 s, counters 2.2 s, speech 45.8 s)`: Layout selection, decoding and scoring the sampled frames, counter OCR, and transcription with diarization. A cached run has no `frames` or `counters` phase. Measured times are in [how long a run takes](#how-long-a-run-takes): about 0.11 x real time frames-only with hardware decode, and 38 min for a 61 min Recording with speech on `large-v3-turbo` (measured with batched decoding, before ADR 0006; about 51 min projected with the sequential decoding now in use), so speech-on runs of a 3 h Recording are far from the 15-20 min target (ADR 0005; the profile and what is left are in issue #30).

The per-second Observations are cached, keyed by a fingerprint of the Recording (its size and its first, middle and last MiB, not its name or mtime) and a hash of the Layout (its content, not its YAML formatting). A rerun with the same Recording and Layout loads them, and the windows that time counter jumps, so it runs no ffmpeg at all. Changing any Layout rectangle or the aspect ratio, or replacing the Recording, misses the cache; an unreadable entry is recomputed. The cache lives in `$XDG_CACHE_HOME/victorious` (`~/.cache/victorious`) unless `--cache-dir` says otherwise, `--no-cache` neither reads nor writes it, and nothing is ever evicted: an entry is about 3-4 MB per 3 h Recording per Layout. A failed run stores no Observations, and a cache that cannot be written is skipped, not an error. A line on stderr reports the Layout chosen with its score and the number of candidates, `cache hit`, `cache miss` or `cache off`, and the wall time.

`--observations observations.jsonl` also writes every raw reading, one JSON line per HUD element per sampled second, in the order `hud_anchor`, `personal_eliminations`, `team_eliminations`, `players_left`, `results_banner`, `placement`, `elimination_placement`, `spectating` (elements the Layout does not describe are left out; `spectating` is always there and scores the label in its fixed top-centre band, read only at seconds with neither the HUD nor a results screen):

| Field | Meaning |
|-------|---------|
| `second` | whole seconds from the start of the Recording |
| `element` | the HUD element |
| `value` | the digits read as an integer, `null` when none were read; only counters and placements have one |
| `confidence` | OCR confidence of `value` (0..1), `0` when `value` is `null` |
| `score` | template match score for `hud_anchor`, `results_banner`, `placement`, `elimination_placement` and `spectating`; `null` for counters |
| `region` | the Layout's `{x, y, w, h}` for the element |

Counters are read only while a Match is open and the HUD is on screen, and only when their pixels changed; any other second has `value: null` for them, and a second that reused the pixels of an earlier one carries that earlier reading. `placement` and `elimination_placement` carry the digits read on the results screen, `value` going to whichever of the two "#" styles matched best. When speech ran and music was masked, one more line per masked span follows the HUD records, `{"element": "music_mask", "start": s, "end": s, "music": m, "speech": p}` (seconds from the start of the Recording; `music` and `speech` are the span's mean per-second scores). The file is written, like `-o`, only after the whole run succeeded; it is not part of the exported JSON schema.

#### Speech

Unless `--no-speech` is given, `analyse` transcribes the audio of every Match window, `[start - 10 s, end + 10 s]` clamped to the Recording, with faster-whisper (model `large-v3-turbo`, int8 on CPU or float16 on an NVIDIA GPU, see [Speech on an NVIDIA GPU](#speech-on-an-nvidia-gpu); see `docs/adr/0005-large-v3-turbo-default-whisper-model.md` for why), decoding each stretch of speech in order with the text before it as context, as whisper.cpp does, which gives punctuated, complete sentences (`docs/adr/0006-sequential-whisper-decoding.md`). It reads roughly 2 s of audio per second on a many-core CPU, so Whisper alone takes about 65-85 min for a 3 h Recording on a quiet machine and 2-3.5 h on a loaded one (measured on one machine; `small` was about nine times faster and clearly less accurate), and it dominates the speech stage. The first audio stream is used; `--audio-stream N` picks another one (counted from 0 among the audio streams). The language is detected per stretch of speech and restricted to English and Portuguese, because Recordings switch between the two; when detection is unsure (the two are within 0.5 of each other), the stretch is decoded in both and the reading with the higher average confidence is kept. Segments whose `no_speech_prob` exceeds 0.9, whose `avg_logprob` is below -1.5 or whose compression ratio exceeds 2.4 are dropped, and so is any run of three or more back-to-back segments with the same words (Whisper's loops on music and noise), so silence and game sound yield no Events. A segment in the margin shared by two Matches goes to the Match whose span is nearer to the middle of the segment (the earlier one on a tie); nothing is duplicated. Audio outside every Match window is never transcribed.

Before transcription, music-dominant audio is silenced so that emote and lobby music is not read as lyrics. A YAMNet audio tagger (ONNX, CPU, about 300 times faster than real time) scores each second of every Match window for `Music` (mean over the frames in that second) and `Speech` (max): a second is music-dominant when music is at least 0.5 and speech below 0.25, so a second with speech over music is never masked. A run of music-dominant seconds bridges one second where the music dipped but nobody spoke, gives up its first or last second when speech is heard right next to it (speech starts and ends inside the second the tagger still calls music), takes in the second on each side where music is still at least 0.25 and nobody speaks, and is masked when what is left lasts at least 3 s (stingers and single beats are left alone). Speech spans are found on the unmodified window audio and then cut at the masked seconds, so a window without a mask is transcribed exactly as before and transcripts outside masks do not move. Overlapping Match margins are tagged once, so a Recording second gets one decision. The masked spans are written to `--observations` as `music_mask` lines and counted on the stderr line (`music masked N s`). The tagger (about 16 MB, pinned to a revision of `audiomagic/yamnet-onnx`) is public and downloaded on first use into the Hugging Face cache. Known limit: vocal music that the tagger scores as `Speech` is not masked and is still transcribed.

#### Speakers

Every `transcript` Event gets a `speaker`. Diarization (pyannote `speaker-diarization-community-1`, on CPU) runs once per Recording over the audio around the transcript segments only, each padded by 1 s and joined with 0.5 s of silence, with music silenced exactly as Whisper saw it (and anything outside the Match windows, which Whisper never saw, silenced too); one run over all Matches keeps a voice's identity across Matches. A segment takes the speaker whose turns overlap it most (the nearest turn when none does). `analyse --voices DIR` (default `./voices`, when it exists) enrolls known Speakers: one clip of a few seconds per person, any audio ffmpeg decodes, named by file stem (`voices/alice.wav` is `alice`). Each diarized speaker is compared with every enrolled voice by the cosine distance of WeSpeaker embeddings (the speaker's mean over its up to 10 longest turns of at least 0.5 s) and takes the nearest name within 0.6; several diarized speakers may share one name. The others are `unknown_1`, `unknown_2`, ... in order of first appearance, stable within one Recording and not across Recordings. A missing or empty `voices/` is not an error (an explicit `--voices` that does not exist adds a stderr warning, and one that is a file exits 1); a clip that cannot be decoded, a stem starting `unknown_`, or two clips with one stem exits 1 before the Recording is read. `speaker` is `null` only when diarization finds no speech at all. The stderr line reports `speakers N (enrolled M)`. The pipeline (about 35 MB, pinned to a revision of the ungated `pyannote-community` mirror, see `docs/adr/0004-ungated-diarization-mirror.md`) is downloaded on first use and needs no token.

#### Categories

Each `transcript` Event is judged personal or game-related by TypeSafe's Jev, asked as one Choice question (`personal` or `game_related`) about the segment text alone; `category` is the option chosen and `category_confidence` is the confidence Jev returns with that answer (distinct from the Whisper `confidence` of the Event). The API key is read from `TYPESAFE_API_KEY`; there is no command-line option for it. `VICTORIOUS_JEV_URL` overrides the endpoint (default `https://api.typesafe.ai/v1/systemone`, model `jev-latest`); it must be `https`, or `http` to `localhost`/`127.0.0.1`/`::1`, and redirects are not followed, so the key is only ever sent to that URL.

Only the text of each segment is sent: no audio, no times, no speaker, no neighbouring segments. The text of casual personal conversation therefore leaves the machine (see `docs/adr/0002-external-categorization-of-transcript-text.md`); transcription and diarization stay local. Passing `--no-categories` or `--no-speech` sends nothing; speech without either needs the key.

Without a key, or when Jev fails, the category is `null` and the run still succeeds with exit 0; there is no heuristic fallback. A key that is unset or blank prints a one-line `note` on stderr when there are segments. Each distinct text is asked once per run, up to 8 requests in parallel with a 30 s timeout; 429, 529, 5xx and network errors are retried (3 attempts in all, 0.5 s then 1 s apart, or the `Retry-After` of the reply when longer, up to 10 s), other 4xx responses and unusable replies leave that one segment `null`. A rejected key (401 or 403), or five failed segments in a row (whatever the failure), stops asking about the rest of the run, which stay `null`, and a `warning` on stderr names the first failure. The stderr line reports `categories K of M`.

Speech needs a Hugging Face token (`HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN` or `hf auth login`); without one `analyse` exits 1 before reading the Recording and says so. The token is only checked for presence here. The audio tagger, the diarization pipeline and the `--voices` clips are loaded (the models downloaded on first use) at the same point, so a missing or corrupt model or clip also fails before the Recording is read. The Whisper model is public: it is downloaded on first use into the Hugging Face cache (about 1.6 GB, with no progress line of ours) and loaded from it afterwards. A Recording without audio, or an `--audio-stream` it does not have, is an error that names `--no-speech`.

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

### Extract

```bash
uv run victorious extract analysis.json --wins -o wins.mkv
```

`extract` reads the JSON `analyse` wrote (an analysis of another schema version exits 1, run `analyse` again), takes every Match with `win: true` and cuts it out of the Recording from `start` to `end + --tail`, where `end` is the first second of the results screen. The tail stops at the start of the next Match and at the end of the Recording. The streams are copied (the codecs of the Recording are kept), so a cut starts at the keyframe at or before `start`, up to one keyframe interval early (about 4 s in a real OBS Recording) and the cuts are then joined, also without re-encoding. With `--exact` each cut is instead re-encoded with libx264 (`-preset veryfast -crf 18`, `yuv420p`) and AAC so it starts on its exact second; that takes about one second per second of footage at 2560x1440 and makes a file of about 12 GB per hour of Wins. The first video stream and every audio stream of the Recording are kept; subtitles and data streams are not. It prints one line per Win on stderr and a summary line at the end. A copied file is about the size of the footage it holds, which is large for a high bitrate Recording (about 16 GB per hour for the one tried).

| Option | Meaning |
|--------|---------|
| `--wins` | required: extract the Wins |
| `-o`, `--output PATH` | required: the video to write; its extension picks the container (`wins.mkv`, `wins.mp4`), an existing file is replaced and it is written whole or not at all |
| `--recording PATH` | the Recording, when it is not at the path written in the JSON (relative to the working directory) |
| `--exact` | re-encode each cut so it starts on its exact second, instead of copying the streams (much slower) |
| `--tail SECONDS` | how long to keep after each results screen appears (default 8) |

It exits 1 with `victorious extract: <reason>` and writes nothing when the analysis has no Win, the Recording is missing, `--output` has no extension, names the Recording or the analysis, or sits in a folder that does not exist, or ffmpeg fails. A missing `--wins` or `-o` is a usage error (exit 2). The scratch files live in a hidden folder beside the output (the cuts take about as much disk as the finished video) and are removed afterwards.

### Layout file

`setup` writes a Layout for you; one can also be written by hand. Keep several in one directory and pass the directory to `--layout`. Coordinates are fractions of the frame (`x`, `y`, `w`, `h` in 0..1), so one Layout serves every resolution of the same HUD design:

```yaml
aspect_ratio: "16:9"
hud_design: icon_column  # or icon_pair or counter_row; optional, icon_column when left out
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

`hud_anchor` must cover the three icons (skull, squad, person) stacked in the HUD column, or the two icons (skull, person) of the `icon_pair` design, which leaves out `team_eliminations` and puts `players_left` on the second row, or in the `counter_row` design the three icons (crosshair, squad, person) of the row under the minimap, where each counter rect is the window right of its icon, as tall as the icon, and the icons are found again at every second because they move with the width of the numbers (see `acceptance/layouts/hud_counter_row_16x9.yaml`); `results_banner` and `placement` locate the Victory Royale banner (blue or orange) and the gold `#N` that goes with it. `elimination_placement`, which is optional, locates the white `YOU PLACED #N` shown when the player is eliminated; it is centred on the frame, so it spans the whole width the `#` can land in. Without it, a "YOU PLACED" screen is not recognised, so only Wins end a Match by a results screen. The three counters are optional and read by later tickets.

See [GLOSSARY.md](GLOSSARY.md) for the terms used above and [docs/adr](docs/adr) for design decisions.

### Speech on an NVIDIA GPU

Whisper and the diarization pipeline run on an NVIDIA GPU through CUDA when one is usable, and on the CPU otherwise; nothing needs to be set up beyond the NVIDIA driver. On Windows `uv sync` installs the CUDA 12.6 build of torch, whose cuBLAS and cuDNN DLLs Whisper (CTranslate2) also loads; on Linux torch stays the CPU build, so there the GPU is used only when a CUDA torch and CUDA 12 libraries are installed by hand.

- A run that uses the GPU prints `victorious: speech: Whisper on the GPU (CUDA)` and `victorious: speech: diarization on the GPU (CUDA)` on stderr. No such line means the CPU.
- On the GPU Whisper decodes in `float16` instead of the CPU's `int8`, and the transcripts are not the same as a CPU run's: on two real 20-30 min Recordings 72-80% of the words matched and the number of segments and Speakers differed, while the Matches and every HUD Event were identical. Two GPU runs of the same Recording also differ a little. Use `VICTORIOUS_DEVICE=cpu` when transcripts must be compared across runs.
- If the GPU cannot load or run a model, one `victorious: ... cannot use the GPU (...); ... on the CPU` line says why and the run goes on on the CPU.
- `VICTORIOUS_DEVICE=cpu` keeps speech on the CPU (use it when comparing runs); `VICTORIOUS_DEVICE=cuda` insists on the GPU and exits 1 naming the reason when it cannot be used.

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
- The speech stack, installed by `uv sync` and not optional ([ADR 0001](docs/adr/0001-mandatory-speech-stack.md)): faster-whisper (Whisper `large-v3-turbo`), pyannote.audio (speaker diarization), torch and torchaudio (CPU builds on Linux, CUDA 12.6 builds on Windows), onnxruntime (YAMNet, which finds music) and huggingface-hub. The install is several GB; `--no-speech` skips the work at run time and never imports these libraries.
- For speech, a Hugging Face token (`HF_TOKEN` or `hf auth login`; every model is public, the token only has to exist) and network on the first speech run, which downloads the models into the Hugging Face cache: Whisper about 1.6 GB, the diarization pipeline about 35 MB and YAMNet about 16 MB. See [step 4](#4-add-speech-transcripts-speakers-categories).
- `ffmpeg` and `ffprobe` on `PATH` (decoding)
- Tesseract, which reads every digit (the three HUD counters and the placements), e.g. `apt install tesseract-ocr`, or on Windows the [UB Mannheim build](https://github.com/UB-Mannheim/tesseract/wiki) with its folder on `PATH`. It provides `libtesseract`, used in the same process, and the `tesseract` command, the fallback and what the tutorial and the tests assume on `PATH`.
- Optional: an NVIDIA GPU for speech (see [Speech on an NVIDIA GPU](#speech-on-an-nvidia-gpu)), an NVIDIA GPU or a VAAPI device for GPU decoding (see [Hardware decode](#hardware-decode)) and `TYPESAFE_API_KEY` for transcript categories (or `--no-categories`).
