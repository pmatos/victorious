# Victorious

Turns long Fortnite gameplay recordings into a timestamped record of what happened in each match, from what is on screen and what is said.

## Language

**Recording**:
A single video file of one play session, typically one to three hours, containing many matches.
_Avoid_: Stream, video, VOD

**Match**:
One Fortnite game, from lobby/bus to results screen.
_Avoid_: Game, round, session

**Win**:
A match whose results screen declares the player victorious (placement #1, Victory Royale).
_Avoid_: Victory, W

**Layout**:
The saved description of where each HUD element sits on screen, for one HUD design at one aspect ratio, in resolution-independent coordinates.
_Avoid_: Config, profile, template

**HUD element**:
A named on-screen item that carries information, such as the personal eliminations counter, team eliminations counter, or players-left counter.
_Avoid_: Widget, region, ROI

**Observation**:
The raw reading of one HUD element at one sampled second, with its confidence, before it is turned into Events.
_Avoid_: Sample, reading, detection

**Event**:
One timestamped fact about a Match or the speech during it, such as match start, an elimination, or the final placement.
_Avoid_: Detection, hit, occurrence

**Setup**:
The one-off step that produces a Layout from a Recording.
_Avoid_: Calibration, training

**Analysis**:
The step that turns a Recording and a Layout into a list of Events.
_Avoid_: Processing, scan

**Transcript segment**:
A span of spoken audio within a Match, with its speaker, its language, and a category of personal or game-related.
_Avoid_: Subtitle, caption

**Music mask**:
A music-dominant span of a Recording's audio that is withheld from transcription, so music is not read as lyrics. Speech over music is not masked.
_Avoid_: Music filter, noise gate

**Speaker**:
A distinct voice in a Recording, named through enrollment or labelled `unknown_N` when not enrolled. Enrollment is one short clip per person in a `voices/` folder, named by file (`voices/alice.wav` is `alice`); `unknown_N` is numbered from 1 by first appearance and is stable only within one Recording.
_Avoid_: Voice, participant
