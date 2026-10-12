"""A deterministic synthetic music bed, and speech mixed over it, written as WAV files."""

import wave
from pathlib import Path

import numpy as np

from victorious.video import AUDIO_RATE, sample_audio

RATE = AUDIO_RATE
MAX_CLIP = 600.0
PEAK = 0.7
CHORDS = (
    (220.0, 277.2, 329.6),
    (246.9, 311.1, 370.0),
    (196.0, 246.9, 293.7),
    (220.0, 277.2, 329.6),
)
BEAT = 0.5


VOWELS = ((800, 1200, 2500), (400, 2000, 2600), (500, 900, 2400), (350, 800, 2400))
PITCHES = (220.0, 247.0, 196.0, 262.0)
SYLLABLE = 0.4
LEAD_GAIN = 0.7


def _lead(seconds: float) -> np.ndarray:
    """A sung vowel line: harmonics shaped by formants, vibrato, a new vowel every syllable.

    Speech detection fires on it, and Whisper reads a phrase into it, yet a tagger still hears
    music because of the chords and drums under it.
    """
    t = np.arange(int(seconds * RATE)) / RATE
    syllable = (t // SYLLABLE).astype(int)
    f0 = np.array(PITCHES)[syllable % len(PITCHES)] * (1 + 0.015 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    out = np.zeros(len(t))
    for harmonic in range(1, 40):
        freq = f0 * harmonic
        gain = np.zeros(len(t))
        for index, formants in enumerate(VOWELS):
            shape = sum(
                weight * np.exp(-0.5 * ((freq - centre) / 150) ** 2)
                for centre, weight in zip(formants, (1.0, 0.7, 0.3), strict=True)
            )
            gain += np.where(syllable % len(VOWELS) == index, shape, 0.0)
        out += gain * np.sin(harmonic * phase) / np.sqrt(harmonic)
    swell = np.clip((0.5 - 0.5 * np.cos(2 * np.pi * (t % SYLLABLE) / SYLLABLE)) * 1.6, 0, 1)
    out *= swell
    return np.asarray(out / np.abs(out).max())


def music_bed(seconds: float, seed: int = 0) -> np.ndarray:
    """Chords, a kick on even beats, a noise hat on odd ones and a sung vowel line, at 0.7 peak."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * RATE))
    for beat in range(int(seconds / BEAT)):
        start = int(beat * BEAT * RATE)
        note = np.arange(min(int(BEAT * RATE * 1.6), len(out) - start)) / RATE
        tone = CHORDS[(beat // 4) % len(CHORDS)][beat % 3]
        harmonics = sum(
            gain * np.sin(2 * np.pi * tone * h * note)
            for h, gain in ((1, 1), (2, 0.5), (3, 0.3), (4, 0.15))
        )
        out[start : start + len(note)] += 0.25 * harmonics * np.exp(-note * 3.0)
        length = min(int(0.15 * RATE), len(out) - start)
        hit = np.arange(length) / RATE
        if beat % 2 == 0:
            out[start : start + length] += (
                0.6 * np.sin(2 * np.pi * (120 * np.exp(-hit * 25)) * hit) * np.exp(-hit * 20)
            )
        else:
            noise = rng.standard_normal(length) * np.exp(-hit * 80)
            out[start : start + length] += 0.15 * noise
    out = out / np.abs(out).max() + LEAD_GAIN * _lead(seconds)[: len(out)]
    return np.asarray(out / np.abs(out).max() * PEAK, dtype="<f4")


def write_wav(path: Path, samples: np.ndarray) -> Path:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm.tobytes())
    return path


def music_with_speech(
    path: Path, speech: Path, *, at: float, seconds: float, speech_gain: float, music_gain: float
) -> Path:
    """`speech` starting `at` seconds into a music bed of `seconds`, mixed at the given gains."""
    mix = music_bed(seconds) * music_gain
    voice = sample_audio(speech, 0.0, MAX_CLIP, 0) * speech_gain
    start = int(at * RATE)
    mix[start : start + len(voice)] += voice[: len(mix) - start]
    return write_wav(path, mix)
