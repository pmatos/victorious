"""Build small, deterministic Recordings from a description of what is on screen when."""

from tests.synthetic.encode import build_recording, decode_frame
from tests.synthetic.layout import layout_document, write_layout
from tests.synthetic.music import music_bed, music_with_speech, write_wav
from tests.synthetic.render import ELEMENT_RECTS
from tests.synthetic.spec import (
    AudioClip,
    Hud,
    Lobby,
    RecordingSpec,
    Results,
    Scene,
    Shot,
    Spectating,
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
    "Spectating",
    "build_recording",
    "decode_frame",
    "layout_document",
    "match_shots",
    "music_bed",
    "music_with_speech",
    "write_layout",
    "write_wav",
]
