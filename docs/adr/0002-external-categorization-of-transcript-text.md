# Transcript text is categorized by an external model; audio stays local

Transcription and diarization run locally, but the personal vs game-related category of each Transcript segment comes from TypeSafe's hosted Jev model, so segment text, including casual personal conversation, leaves the machine. We accepted this because Jev returns calibrated typed judgments that a keyword heuristic cannot match, and we kept audio local so the exposure is limited to text. With no API key the category is `null`, with no heuristic fallback.
