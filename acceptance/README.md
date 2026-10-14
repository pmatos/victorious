# Acceptance data (#30)

The Layout and labels files used to check `analyse` on three real Recordings (`2026-10-08_11-11-58`, `2026-10-09_11-59-53`, `2026-10-12_12-04-04`, 2560x1440 AV1, 61 to 83 min). The Recordings themselves are not in the repository.

- `layouts/hud_icons_16x9.yaml`: the Layout made by `setup` for the icon-column HUD; it scores 0.98 on all three Recordings.
- `labels/<stem>.labels.json`: one labels file per Recording.

**The labels were produced by an automated visual check, not by the owner.** The detected Matches were compared with frames of the Recording (start, last in-match frame, results screen), and the labels written from that. They need the owner's confirmation through `victorious review`, which rewrites the file with the owner's verdicts.

Reproduce (put the Recordings anywhere; only the file name is compared):

```bash
uv run victorious analyse --no-speech --no-cache --layout acceptance/layouts \
    -o analysis.json /path/to/2026-10-09_11-59-53.mkv
uv run victorious review analysis.json --against acceptance/labels/2026-10-09_11-59-53.labels.json
```

Notes on what the labels contain:

- `2026-10-12_12-04-04` has two short Matches (325 to 469 s and 1718 to 1765 s) that the player left before a results screen; they have no placement and are labelled `confirmed` as Matches.
- Wins: two in `2026-10-09_11-59-53`, two in `2026-10-12_12-04-04` (one of them, ending at 2985 s, shows the orange Victory Royale banner under the Social overlay); none in `2026-10-08_11-11-58`.
- Only the icon-column HUD is covered. A Recording with the horizontal-row HUD (#48) is not in this set.
