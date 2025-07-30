# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Victorious is a Python tool for detecting when specific images appear in video frames using OpenCV template matching. The application processes videos to find temporal occurrences of a template image and outputs the timestamps when matches are detected.

## Architecture

- **Single-module design**: The entire application is contained in `main.py`
- **Command-line interface**: Uses argparse for parameter handling
- **Computer vision pipeline**: OpenCV-based template matching with configurable threshold
- **Frame processing**: Sequential video frame analysis with detection state tracking

## Development Commands

### Dependency Management
```bash
# Install dependencies (uses uv)
uv sync

# Install development dependencies
uv sync --group dev
```

### Code Quality
```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Fix linting issues
uv run ruff check --fix

# Type checking
uv run mypy main.py
```

### Running the Application
```bash
# Basic usage
uv run python main.py <video_path> <image_path>

# With custom threshold
uv run python main.py <video_path> <image_path> --threshold 0.9
```

## Code Structure

The application follows a functional design with clear separation of concerns:

- `parse_arguments()`: CLI argument processing
- `load_template()`: Image loading with error handling
- `match_template_in_frame()`: Core template matching logic
- `detect_image_in_video()`: Video processing pipeline with state tracking
- `main()`: Entry point with comprehensive error handling

## Configuration

- **Ruff**: Configured for 100-character line length with Python 3.11+ target
- **Dependencies**: Uses modern Python versions (>=3.13) with OpenCV and NumPy
- **Linting**: Enforces error checking, pyupgrade, bugbear, simplify, and import sorting rules