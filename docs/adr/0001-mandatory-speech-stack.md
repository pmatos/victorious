# Speech dependencies are installed by default, not an optional extra

Transcription (faster-whisper), diarization (pyannote.audio, a gated Hugging Face model) and audio tagging pull in torch and several GB of models, which makes the install heavy for a tool whose core is frame analysis. We chose to make them mandatory because Recordings are always analysed with their audio, and one install path is simpler than two feature tiers. `--no-speech` still skips the work at run time, and a missing Hugging Face token fails fast at startup.
