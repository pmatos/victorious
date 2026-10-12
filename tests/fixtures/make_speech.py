"""Regenerate the two committed speech clips with Piper text-to-speech.

Run by hand from the repository root when a clip needs refreshing; CI never runs it. Needs
`uvx`, `ffmpeg` and the two Piper voices downloaded next to each other (the .onnx and its
.onnx.json) from https://huggingface.co/rhasspy/piper-voices:

    VICTORIOUS_PIPER_VOICES=/path/to/voices uv run python -m tests.fixtures.make_speech

English uses en_US-ljspeech-medium (public domain data); Portuguese uses pt_BR-faber-medium
(CC0 data, fine-tuned from lessac). The sentences are ours.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

OUT = Path(__file__).parent / "speech"
CLIPS = {
    "en": (
        "en_US-ljspeech-medium",
        "Nice shot, there are only ten players left, let's rotate to the zone.",
    ),
    "pt": (
        "pt_BR-faber-medium",
        "Boa jogada, só restam dez jogadores, vamos rodar para a zona agora.",
    ),
}


def main() -> int:
    voices = os.environ.get("VICTORIOUS_PIPER_VOICES")
    if voices is None:
        print("set VICTORIOUS_PIPER_VOICES to the directory holding the Piper voices")
        return 2
    uvx, ffmpeg = shutil.which("uvx"), shutil.which("ffmpeg")
    if uvx is None or ffmpeg is None:
        print("uvx and ffmpeg must be on PATH")
        return 2
    OUT.mkdir(exist_ok=True)
    for name, (voice, sentence) in CLIPS.items():
        with tempfile.TemporaryDirectory() as scratch:
            wav = Path(scratch) / f"{name}.wav"
            subprocess.run(
                [
                    uvx, "--from", "piper-tts", "python", "-m", "piper",
                    "-m", str(Path(voices) / f"{voice}.onnx"), "-f", str(wav),
                ],
                input=sentence,
                text=True,
                check=True,
            )  # fmt: skip
            subprocess.run(
                [
                    ffmpeg, "-v", "error", "-y", "-i", str(wav), "-ac", "1", "-ar", "16000",
                    "-c:a", "libopus", "-b:a", "24k", str(OUT / f"{name}.opus"),
                ],
                check=True,
            )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
