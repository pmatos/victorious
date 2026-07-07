# Victorious

A Python tool for detecting when specific images appear in video frames using OpenCV template matching.

## Features

- Detect appearances of a template image within video frames
- Configurable matching threshold for precision control
- Output timestamps of detected appearances
- Handles various video and image formats supported by OpenCV

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install dependencies with:

```bash
uv sync
```

## Usage

### Basic Usage

```bash
uv run python main.py <video_path> <image_path>
```

### With Custom Threshold

```bash
uv run python main.py <video_path> <image_path> --threshold 0.9
```

### Parameters

- `video_path`: Path to the video file to analyze
- `image_path`: Path to the template image to detect
- `--threshold`: Matching threshold (0.0-1.0, default: 0.8)
  - Higher values require more precise matches
  - Lower values allow more lenient matching

### Example

```bash
uv run python main.py video.mp4 logo.png --threshold 0.85
```

This will output timestamps (in seconds) when the logo appears in the video:

```
12.34
45.67
78.90
```

If no matches are found, it outputs:

```
No matches found
```

## How It Works

1. **Template Loading**: The specified image is loaded as a template
2. **Video Processing**: Each frame of the video is analyzed sequentially
3. **Template Matching**: OpenCV's `cv2.matchTemplate()` with normalized correlation coefficient
4. **Detection Logic**: Tracks detection state to output only the start of each appearance
5. **Timestamp Calculation**: Converts frame numbers to timestamps using video FPS

## Development

### Code Quality

```bash
# Format code
uv run ruff format

# Lint code
uv run ruff check

# Type checking
uv run mypy
```

### Testing

```bash
# Run the test suite
uv run pytest

# Run with coverage
uv run pytest --cov=. --cov-report=term-missing
```

## Requirements

- Python >= 3.13
- OpenCV (opencv-python)
- NumPy

## License

This project is open source. See the project files for license details.