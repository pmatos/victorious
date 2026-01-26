# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victorious is a Python tool for detecting when specific images appear in video frames using OpenCV template matching. It processes videos to find temporal occurrences of a template image and outputs timestamps when matches are detected.

## Architecture

**Single-module design**: The entire application is contained in `main.py` with these key processing modes:

- **Fast mode** (default): Extracts frames to disk using ffmpeg at configurable FPS/resolution, then processes in parallel. Best for long videos.
- **Direct mode**: Reads frames via OpenCV VideoCapture with multi-backend fallback. Falls back to chunked processing if needed.
- **Chunked mode**: Splits video into segments for processing problematic formats.

**Template matching pipeline**:
- Multi-scale matching tests templates at scales 0.3-1.2 (0.1 increments) to handle resolution differences
- Grayscale matching (default in fast mode) for ~3x speedup
- State tracking to output only the start timestamp of each detection

## Development Commands

```bash
# Install dependencies
uv sync

# Install dev dependencies
uv sync --group dev

# Format and lint
uv run ruff format
uv run ruff check
uv run ruff check --fix

# Type checking
uv run mypy main.py

# Run application
uv run python main.py <video_path> <image_path>
uv run python main.py video.mkv template.png --threshold 0.8 --fps 2 --max-height 720
```

## Key CLI Options

- `--threshold`: Match confidence (0.0-1.0, default: 0.6)
- `--fps`: Frame extraction rate for fast mode (default: 2.0)
- `--max-height`: Max frame height for fast mode (default: 720)
- `--grayscale/--no-grayscale`: Grayscale matching (default: enabled)
- `--fast-mode/--no-fast-mode`: ffmpeg frame extraction (default: enabled)
- `--scales`: Comma-separated scale factors for multi-scale matching

## Configuration

- **Ruff**: 100-character line length, Python 3.11+ target
- **Dependencies**: Python >=3.13, OpenCV, NumPy
