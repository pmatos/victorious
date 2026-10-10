"""Build small, deterministic Recordings from a description of what is on screen when."""

from tests.synthetic.encode import build_recording
from tests.synthetic.render import ELEMENT_RECTS
from tests.synthetic.spec import (
    AudioClip,
    Hud,
    Lobby,
    RecordingSpec,
    Results,
    Scene,
    Shot,
    match_shots,
)

__all__ = [
    "ELEMENT_RECTS",
    "AudioClip",
    "Hud",
    "Lobby",
    "RecordingSpec",
    "Results",
    "Scene",
    "Shot",
    "build_recording",
    "match_shots",
]
