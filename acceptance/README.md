# Acceptance data (#30)

The Layout and labels files used to check `analyse` on three real Recordings (`2026-10-08_11-11-58`, `2026-10-09_11-59-53`, `2026-10-12_12-04-04`, 2560x1440 AV1, 61 to 83 min). The Recordings themselves are not in the repository.

- `layouts/hud_icons_16x9.yaml`: the Layout made by `setup` for the icon-column HUD; it scores 0.98 on all three Recordings.
- `layouts/hud_counter_row_16x9.yaml`: the Layout made by `setup` for the second HUD design (#48), the three counters in a horizontal row of icons under the minimap (`hud_design: counter_row`). It was made from `test-video-last-minute.mkv`, a 60 s 2560x1440 clip that is the last minute of a Match (not in the repository), with `victorious setup test-video-last-minute.mkv --at hud=5 --at results=38 -o hud_counter_row_16x9.yaml`; it scores 0.93 on the clip. It has no `elimination_placement` because the clip has no "YOU PLACED" screen.
- `labels/test-video-last-minute.labels.json`: the one Match of that clip (0 to 34 s, a Win with placement 1, 10 personal eliminations; the counters read 7 / 9 / 24 for players left / personal / team eliminations until 13 s and 6 / 10 / 25 after).
- `labels/<stem>.labels.json`: one labels file per Recording.

**The labels were produced by an automated visual check, not by the owner.** The detected Matches were compared with frames of the Recording (start, last in-match frame, results screen), and the labels written from that. They need the owner's confirmation through `victorious review`, which rewrites the file with the owner's verdicts.

Reproduce (put the Recordings anywhere; only the file name is compared):

```bash
uv run victorious analyse --no-speech --no-cache --layout acceptance/layouts \
    -o analysis.json /path/to/2026-10-09_11-59-53.mkv
uv run victorious review analysis.json --against acceptance/labels/2026-10-09_11-59-53.labels.json
```

To confirm the labels of a Recording, run the `analyse` command above, then serve the review page and write the verdicts over the committed file:

```bash
uv run victorious review analysis.json --recording /path/to/2026-10-09_11-59-53.mkv \
    -o acceptance/labels/2026-10-09_11-59-53.labels.json
```

A page opens (the address is on stderr; `--no-browser` only prints it) with one card per Match and two frames of the Recording. Confirm each Match whose Win, placement and personal eliminations are right and reject the rest (`c` and `r` decide the focused card); `Save labels` is enabled once every Match has a verdict. `-o` is needed because the default is `analysis.labels.json` beside the JSON, and `--recording` because the JSON holds the path the Recording had when `analyse` ran (without it, `review` exits with an error when the Recording is not at that path). The new file replaces the old one, so check `git diff acceptance/labels` before committing: a rejected Match or a changed value is a disagreement between the owner and the automated check. The file also records the `analysis.schema_version` of the run that produced it (`1.6` in the current files, `1.7` now), which `--against` does not read.

Notes on what the labels contain:

- `2026-10-12_12-04-04` has two short Matches (325 to 469 s and 1718 to 1765 s) that the player left before a results screen; they have no placement and are labelled `confirmed` as Matches.
- Wins: two in `2026-10-09_11-59-53`, two in `2026-10-12_12-04-04` (one of them, ending at 2985 s, shows the orange Victory Royale banner under the Social overlay); none in `2026-10-08_11-11-58`.
- The icon-column HUD is covered by the three Recordings; the counter-row HUD only by the 60 s clip (one Match, no eliminated results screen, two-digit counters), so its counter reading is verified on that clip and on synthetic Recordings, not on a full Recording.
- With a folder holding both Layouts, `analyse` picks the counter-row Layout for the clip and the icon-column Layout for the three Recordings (by the score of each Layout's `hud_anchor`). With only the icon-column Layout, the clip is refused with `no Layout matches this Recording's HUD` (exit 1), where it used to be reported as `not_fortnite`.
- Labels do not cover `eliminated` Events (#53). On `2026-10-08_11-11-58` the third Match (2653 to 3817 s) has one at about 3534 s (the player is out and spectating, the squad plays on; a teammate reboots the player at about 3630 s) and one at 3817 s with placement 2 (the "YOU PLACED #2" screen that ends the Match). Every non-Win Match ends on such a screen straight after its in-match HUD, so the squad's placement and the player's own elimination coincide there.
