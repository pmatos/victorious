import argparse
import contextlib
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from pathlib import Path
from types import FrameType
from typing import Any

import cv2
import numpy as np

# Global flags
DEBUG = False
INTERRUPTED = False


def debug_log(message: str) -> None:
    """Print debug message if debug mode is enabled."""
    if DEBUG:
        print(f"DEBUG: {message}", file=sys.stderr)


def signal_handler(signum: int, frame: FrameType | None) -> None:
    """Handle interrupt signal gracefully."""
    global INTERRUPTED
    INTERRUPTED = True
    print("\nProcessing interrupted by user. Cleaning up...", file=sys.stderr)


def parse_scales(scales_str: str) -> list[float]:
    """Parse a comma-separated string of scale factors into a list of floats.

    Empty entries (e.g. trailing commas) are ignored. Raises ValueError if a
    non-empty entry is not a valid float.
    """
    return [float(part) for part in scales_str.split(",") if part.strip()]


def frame_index_to_timestamp(frame_index: int, fps: float) -> float:
    """Convert a zero-based frame index to a timestamp in seconds."""
    return frame_index / fps


def extract_detection_times(matches: Iterable[tuple[bool, float]]) -> list[float]:
    """Reduce per-frame match results to the start timestamp of each run.

    Given ``(is_match, timestamp)`` pairs in chronological order, emit the
    timestamp only when a match run begins (a False->True transition, or the
    first frame if it already matches). Consecutive matches collapse to a single
    start; a run only restarts after at least one non-matching frame.
    """
    detection_times: list[float] = []
    currently_detected = False
    for is_match, timestamp in matches:
        if is_match and not currently_detected:
            detection_times.append(timestamp)
            currently_detected = True
        elif not is_match and currently_detected:
            currently_detected = False
    return detection_times


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect image appearances in video frames")
    parser.add_argument("video_path", type=Path, help="Path to the video file")
    parser.add_argument("image_path", type=Path, help="Path to the image to detect")
    parser.add_argument(
        "--threshold", type=float, default=0.6, help="Matching threshold (0.0-1.0, default: 0.6)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=60,
        help="Chunk size in seconds for large videos (default: 60)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Process every Nth frame (default: 1, process all frames)",
    )
    parser.add_argument(
        "--scales",
        type=str,
        default="0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2",
        help="Comma-separated list of scales to test (default: 0.3-1.2 in 0.1 steps)",
    )
    parser.add_argument(
        "--single-scale",
        action="store_true",
        help="Use single-scale matching only (faster but less accurate)",
    )
    parser.add_argument(
        "--cores",
        type=int,
        default=0,
        help="Number of CPU cores to use for processing (0 = use all cores, default: 0)",
    )
    parser.add_argument(
        "--force-chunking",
        action="store_true",
        help="Force chunked processing even if direct processing might work",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frame extraction rate for fast mode (default: 2.0 fps)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=720,
        help="Maximum frame height for fast mode (default: 720)",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        default=True,
        help="Use grayscale matching for faster processing (default: enabled)",
    )
    parser.add_argument(
        "--no-grayscale",
        action="store_false",
        dest="grayscale",
        help="Disable grayscale (slower but more accurate for color-specific templates)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        default=True,
        help="Use fast frame extraction mode with ffmpeg (default: enabled)",
    )
    parser.add_argument(
        "--no-fast-mode",
        action="store_false",
        dest="fast_mode",
        help="Disable fast mode, use traditional frame-by-frame processing",
    )
    return parser.parse_args()


def load_template(image_path: Path, grayscale: bool = False) -> np.ndarray:
    if grayscale:
        template = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    else:
        template = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"Could not load image: {image_path}")
    return template


def match_template_in_frame_multiscale(
    frame: np.ndarray,
    template: np.ndarray,
    threshold: float,
    scales: list[float] | None = None,
) -> tuple[bool, float, float]:
    """
    Multi-scale template matching.
    Returns: (is_match, best_confidence, best_scale)
    """
    if scales is None:
        scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    template_h, template_w = template.shape[:2]
    frame_h, frame_w = frame.shape[:2]

    best_confidence = 0.0
    best_scale = 1.0

    for scale in scales:
        # Calculate scaled template dimensions
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)

        # Skip if template would be too large for frame
        if new_w > frame_w or new_h > frame_h:
            debug_log(
                f"Skipping scale {scale:.2f}x - template too large "
                f"({new_w}x{new_h} > {frame_w}x{frame_h})"
            )
            continue

        # Skip if template would be too small to be meaningful
        if new_w < 10 or new_h < 10:
            debug_log(f"Skipping scale {scale:.2f}x - template too small ({new_w}x{new_h})")
            continue

        try:
            # Resize template
            scaled_template = cv2.resize(template, (new_w, new_h))

            # Perform template matching
            result = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)

            debug_log(f"Scale {scale:.2f}x: confidence {max_val:.4f}")

            # Update best match
            if max_val > best_confidence:
                best_confidence = max_val
                best_scale = scale

                # Early termination for very high confidence matches
                if max_val > 0.95:
                    debug_log(
                        f"Early termination at scale {scale:.2f}x with confidence {max_val:.4f}"
                    )
                    break

        except cv2.error as e:
            debug_log(f"OpenCV error at scale {scale:.2f}x: {e}")
            continue

    is_match = best_confidence >= threshold
    debug_log(
        f"Multi-scale result: best_confidence={best_confidence:.4f}, "
        f"best_scale={best_scale:.2f}x, is_match={is_match}"
    )

    return is_match, best_confidence, best_scale


def match_template_in_frame(frame: np.ndarray, template: np.ndarray, threshold: float) -> bool:
    """
    Backward compatibility wrapper for single-scale template matching.
    """
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= threshold


def process_frame_worker(args: tuple[Any, ...]) -> tuple[int, float, bool, float, float]:
    """
    Worker functions for multiprocessing frame analysis.
    Returns: (frame_index, timestamp, is_match, confidence, best_scale)
    """
    frame_index, frame_data, template, threshold, scales, use_single_scale, fps = args

    timestamp = frame_index_to_timestamp(frame_index, fps)

    if use_single_scale:
        is_match = match_template_in_frame(frame_data, template, threshold)
        confidence = 0.0  # Not calculated in single-scale mode
        best_scale = 1.0
    else:
        is_match, confidence, best_scale = match_template_in_frame_multiscale(
            frame_data, template, threshold, scales
        )

    return frame_index, timestamp, is_match, confidence, best_scale


def process_frame_batch(
    frame_batch: list[tuple[int, np.ndarray]],
    template: np.ndarray,
    threshold: float,
    scales: list[float] | None,
    use_single_scale: bool,
    fps: float,
    num_cores: int,
) -> list[tuple[int, float, bool, float, float]]:
    """
    Process a batch of frames using multiprocessing.
    Returns list of (frame_index, timestamp, is_match, confidence, best_scale) tuples.
    """
    if num_cores == 1:
        # Single-threaded processing
        results = []
        for frame_index, frame_data in frame_batch:
            args = (frame_index, frame_data, template, threshold, scales, use_single_scale, fps)
            results.append(process_frame_worker(args))
        return sorted(results, key=lambda x: x[0])  # Sort by frame_index

    # Multi-threaded processing
    if scales is None:
        scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    # Prepare arguments for worker processes
    worker_args = [
        (frame_index, frame_data, template, threshold, scales, use_single_scale, fps)
        for frame_index, frame_data in frame_batch
    ]

    # Use multiprocessing pool
    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(process_frame_worker, worker_args)

    # Sort results by frame index to maintain chronological order
    return sorted(results, key=lambda x: x[0])


def is_video_readable(video_path: Path, test_frames: int = 5) -> bool:
    """Test if OpenCV can properly read video frames."""
    import contextlib
    import io

    # Capture stderr to avoid spam during testing
    stderr_capture = io.StringIO()

    with contextlib.redirect_stderr(stderr_capture):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False

        valid_frames = 0
        for _ in range(test_frames):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                valid_frames += 1
                # If we get a couple good frames quickly, it's probably working
                if valid_frames >= 2:
                    break
            else:
                break

        cap.release()

    # Check if we got meaningful errors indicating codec issues
    stderr_content = stderr_capture.getvalue()
    if (
        "Missing Sequence Header" in stderr_content
        or "Failed to get pixel format" in stderr_content
    ):
        return False

    return valid_frames >= 2


def convert_video_to_h264(input_path: Path) -> Path | None:
    """Convert video to H.264 format using FFmpeg."""
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4", prefix="victorious_")
        os.close(temp_fd)

        # Check video size/duration first
        probe_cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(input_path),
        ]
        try:
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
            duration = float(result.stdout.strip()) if result.stdout.strip() else 0
        except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
            duration = 0

        cmd = [
            "ffmpeg",
            "-v",
            "warning",
            "-i",
            str(input_path),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-f",
            "mp4",
        ]

        cmd.extend(["-y", temp_path])

        # Allow more time for longer videos (0.5s per 1s of video, min 60s)
        timeout = max(60, duration * 0.5) if duration > 0 else 300
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        return Path(temp_path)

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        if "temp_path" in locals() and Path(temp_path).exists():
            Path(temp_path).unlink()

        # isinstance checks dispatch on the caught exception, not on user input,
        # so ValueError (the caller's contract) is correct here despite TRY004.
        if isinstance(e, FileNotFoundError):
            raise ValueError(  # noqa: TRY004
                "FFmpeg not found. Please install FFmpeg to process this video format."
            ) from e
        elif isinstance(e, subprocess.TimeoutExpired):
            raise ValueError(  # noqa: TRY004
                "Video conversion timed out. The video may be too large or complex."
            ) from e
        else:
            raise ValueError(f"Video conversion failed: {e.stderr}") from e  # noqa: TRY004


def cleanup_temp_file(temp_path: Path | None) -> None:
    """Clean up temporary file if it exists."""
    if temp_path and temp_path.exists():
        with contextlib.suppress(OSError):
            temp_path.unlink()


def cleanup_temp_dir(temp_dir: Path | None) -> None:
    """Clean up temporary directory and all its contents."""
    if temp_dir and temp_dir.exists():
        import shutil

        with contextlib.suppress(OSError):
            shutil.rmtree(temp_dir)


def extract_frames_to_dir(
    video_path: Path,
    fps: float = 2.0,
    max_height: int = 720,
) -> tuple[Path, int]:
    """
    Extract frames from video using ffmpeg at specified FPS and resolution.
    Returns: (temp_directory_path, number_of_frames_extracted)
    """
    debug_log(f"Extracting frames at {fps} fps, max height {max_height}px")

    temp_dir = Path(tempfile.mkdtemp(prefix="victorious_frames_"))
    frame_pattern = temp_dir / "frame_%06d.jpg"

    # Build ffmpeg command with scale filter
    # -1:max_height maintains aspect ratio
    scale_filter = f"fps={fps},scale=-1:{max_height}:flags=fast_bilinear"

    cmd = [
        "ffmpeg",
        "-v",
        "warning",
        "-i",
        str(video_path),
        "-vf",
        scale_filter,
        "-q:v",
        "2",  # High quality JPEG
        "-an",  # No audio
        str(frame_pattern),
    ]

    try:
        debug_log(f"Running: {' '.join(cmd[:8])}...")
        start_time = time.time()

        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        extraction_time = time.time() - start_time

        # Count extracted frames
        frame_files = sorted(temp_dir.glob("frame_*.jpg"))
        num_frames = len(frame_files)

        debug_log(
            f"Extracted {num_frames} frames in {extraction_time:.1f}s "
            f"({num_frames / extraction_time:.1f} frames/s)"
        )

        if num_frames == 0:
            cleanup_temp_dir(temp_dir)
            raise ValueError("No frames were extracted from the video")

    except subprocess.CalledProcessError as e:
        cleanup_temp_dir(temp_dir)
        raise ValueError(f"Frame extraction failed: {e.stderr}") from e
    except FileNotFoundError as e:
        cleanup_temp_dir(temp_dir)
        raise ValueError("FFmpeg not found. Please install FFmpeg to use fast mode.") from e
    else:
        return temp_dir, num_frames


def process_frame_file_worker(args: tuple[Any, ...]) -> tuple[int, float, bool, float, float]:
    """
    Worker function for processing a single frame file from disk.
    Returns: (frame_index, timestamp, is_match, confidence, best_scale)
    """
    frame_path, frame_index, template, threshold, scales, use_single_scale, fps, use_grayscale = (
        args
    )

    timestamp = frame_index_to_timestamp(frame_index, fps)

    # Load frame from disk
    if use_grayscale:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    else:
        frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

    if frame is None:
        return frame_index, timestamp, False, 0.0, 1.0

    if use_single_scale:
        is_match = match_template_in_frame(frame, template, threshold)
        confidence = 0.0
        best_scale = 1.0
    else:
        is_match, confidence, best_scale = match_template_in_frame_multiscale(
            frame, template, threshold, scales
        )

    return frame_index, timestamp, is_match, confidence, best_scale


def process_extracted_frames(
    frames_dir: Path,
    template: np.ndarray,
    threshold: float,
    fps: float,
    scales: list[float] | None = None,
    use_single_scale: bool = False,
    num_cores: int = 0,
    use_grayscale: bool = True,
) -> list[float]:
    """
    Process pre-extracted frame images using multiprocessing.
    Returns list of detection timestamps.
    """
    debug_log(f"Processing extracted frames from {frames_dir}")

    # Get sorted list of frame files
    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    num_frames = len(frame_files)

    if num_frames == 0:
        return []

    # Determine number of cores
    if num_cores <= 0:
        num_cores = multiprocessing.cpu_count()

    debug_log(f"Processing {num_frames} frames using {num_cores} cores")

    if scales is None:
        scales = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    # Prepare worker arguments
    worker_args = [
        (frame_path, idx, template, threshold, scales, use_single_scale, fps, use_grayscale)
        for idx, frame_path in enumerate(frame_files)
    ]

    # Process frames in parallel
    start_time = time.time()
    last_progress_time = start_time

    if num_cores == 1:
        results = []
        for i, args in enumerate(worker_args):
            if INTERRUPTED:
                break
            results.append(process_frame_file_worker(args))

            # Progress updates
            current_time = time.time()
            if current_time - last_progress_time > 10:
                progress = (i + 1) / num_frames * 100
                elapsed = current_time - start_time
                print(
                    f"Processed {i + 1}/{num_frames} frames "
                    f"({progress:.1f}%) - {elapsed:.1f}s elapsed",
                    file=sys.stderr,
                )
                last_progress_time = current_time
    else:
        # Use multiprocessing pool with progress tracking
        with multiprocessing.Pool(processes=num_cores) as pool:
            results = []
            chunk_size = max(1, num_frames // (num_cores * 4))

            for i, result in enumerate(
                pool.imap_unordered(process_frame_file_worker, worker_args, chunksize=chunk_size)
            ):
                if INTERRUPTED:
                    pool.terminate()
                    break
                results.append(result)

                # Progress updates every 10 seconds
                current_time = time.time()
                if current_time - last_progress_time > 10:
                    progress = (i + 1) / num_frames * 100
                    elapsed = current_time - start_time

                    if i > 100:
                        frames_per_sec = (i + 1) / elapsed
                        remaining = (num_frames - i - 1) / frames_per_sec
                        remaining_min = int(remaining // 60)
                        remaining_sec = int(remaining % 60)
                        print(
                            f"Processed {i + 1}/{num_frames} frames ({progress:.1f}%) - "
                            f"{elapsed:.1f}s elapsed, ~{remaining_min}m {remaining_sec}s remaining",
                            file=sys.stderr,
                        )
                    else:
                        print(
                            f"Processed {i + 1}/{num_frames} frames "
                            f"({progress:.1f}%) - {elapsed:.1f}s elapsed",
                            file=sys.stderr,
                        )
                    last_progress_time = current_time

    # Sort results by frame index
    results = sorted(results, key=lambda x: x[0])

    # Apply state tracking to get detection times
    detection_times = extract_detection_times(
        (is_match, timestamp) for _, timestamp, is_match, _, _ in results
    )
    for timestamp in detection_times:
        debug_log(f"Match found at timestamp {timestamp:.2f}s")

    process_time = time.time() - start_time
    frames_per_sec = num_frames / process_time if process_time > 0 else 0

    print(
        f"Frame analysis completed in {process_time:.1f}s ({frames_per_sec:.1f} frames/s). "
        f"Found {len(detection_times)} matches.",
        file=sys.stderr,
    )

    return detection_times


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return float(result.stdout.strip()) if result.stdout.strip() else 0
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired):
        return 0


def get_video_codec(video_path: Path) -> str:
    """Get video codec name using ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        codec = result.stdout.strip().lower()
        debug_log(f"Detected video codec: {codec}")
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        debug_log("Could not detect video codec, will use re-encoding")
        return ""
    else:
        return codec


def get_video_properties(video_path: Path) -> dict[str, str]:
    """Get video properties (resolution, fps, codec) using ffprobe."""
    properties = {"width": "unknown", "height": "unknown", "fps": "unknown", "codec": "unknown"}

    try:
        # Get resolution and fps
        cmd = [
            "ffprobe",
            "-v",
            "quiet",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,codec_name",
            "-of",
            "json",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            import json

            data = json.loads(result.stdout)
            if data.get("streams"):
                stream = data["streams"][0]
                properties["width"] = str(stream.get("width", "unknown"))
                properties["height"] = str(stream.get("height", "unknown"))
                properties["codec"] = stream.get("codec_name", "unknown")

                # Parse frame rate
                fps_str = stream.get("r_frame_rate", "")
                if fps_str and "/" in fps_str:
                    num, denom = fps_str.split("/")
                    properties["fps"] = f"{int(num) / int(denom):.2f}"
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
        pass

    return properties


def create_video_chunk(
    input_path: Path, start_time: float, duration: float, chunk_index: int
) -> Path:
    """Create a temporary video chunk for processing."""
    debug_log(f"Creating chunk {chunk_index} from {start_time:.1f}s for {duration:.1f}s")
    start_chunk_time = time.time()

    temp_fd, temp_path = tempfile.mkstemp(suffix=f"_chunk_{chunk_index}.mp4", prefix="victorious_")
    os.close(temp_fd)

    # Detect codec to decide between stream copy (fast) and re-encoding (slow)
    codec = get_video_codec(input_path)
    compatible_codecs = ["h264", "h265", "hevc", "vp8", "vp9", "mpeg4"]

    # Use stream copy for compatible codecs (400x faster)
    if codec in compatible_codecs:
        debug_log(f"Using stream copy for codec {codec} (fast)")
        cmd = [
            "ffmpeg",
            "-v",
            "warning",
            "-ss",
            str(start_time),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c",
            "copy",  # Stream copy - no re-encoding
            "-avoid_negative_ts",
            "make_zero",
            "-f",
            "mp4",
            "-y",
            temp_path,
        ]
    else:
        # Re-encode for problematic codecs (AV1, etc.)
        debug_log(f"Re-encoding for codec {codec} (slow)")
        cmd = [
            "ffmpeg",
            "-v",
            "warning",
            "-ss",
            str(start_time),
            "-i",
            str(input_path),
            "-t",
            str(duration),
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "28",
            "-avoid_negative_ts",
            "make_zero",
            "-f",
            "mp4",
            "-y",
            temp_path,
        ]

    try:
        debug_log(f"Running FFmpeg command: {' '.join(cmd[:8])}...")
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)

        chunk_time = time.time() - start_chunk_time
        debug_log(f"Chunk {chunk_index} created in {chunk_time:.2f}s")

        return Path(temp_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if Path(temp_path).exists():
            Path(temp_path).unlink()
        debug_log(f"Chunk creation failed: {e}")
        raise ValueError(f"Failed to create video chunk: {e}") from e


def process_video_in_chunks(
    video_path: Path,
    template: np.ndarray,
    threshold: float,
    chunk_duration: float = 60.0,
    frame_skip: int = 1,
    scales: list[float] | None = None,
    use_single_scale: bool = False,
    num_cores: int = 0,
) -> list[float]:
    """Process video in chunks with progress updates."""
    debug_log(f"Starting chunked processing with {chunk_duration}s chunks")
    total_start_time = time.time()

    total_duration = get_video_duration(video_path)
    debug_log(f"Video duration: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")

    if total_duration <= 0:
        debug_log("Could not determine video duration, falling back to direct processing")
        return detect_image_in_video_direct(
            video_path, template, threshold, frame_skip, scales, use_single_scale, num_cores
        )

    print(
        f"Processing {total_duration / 60:.1f} minute video in {chunk_duration:.0f}s chunks...",
        file=sys.stderr,
    )

    detection_times: list[float] = []
    chunk_paths = []
    num_chunks = int((total_duration + chunk_duration - 1) // chunk_duration)  # Ceiling division
    debug_log(f"Will create {num_chunks} chunks")

    chunk_times: list[float] = []  # Track time per chunk for estimation

    try:
        for chunk_index in range(num_chunks):
            # Check for interruption
            if INTERRUPTED:
                debug_log(f"Processing interrupted at chunk {chunk_index + 1}/{num_chunks}")
                print(
                    f"Processed {chunk_index}/{num_chunks} chunks before interruption. "
                    f"Found {len(detection_times)} matches.",
                    file=sys.stderr,
                )
                break

            chunk_start_time = time.time()
            start_time = chunk_index * chunk_duration
            remaining_time = total_duration - start_time
            actual_duration = min(chunk_duration, remaining_time)

            progress = (chunk_index + 1) / num_chunks * 100
            elapsed = time.time() - total_start_time

            # Calculate time remaining estimate after first chunk
            if chunk_index > 0:
                avg_chunk_time = sum(chunk_times) / len(chunk_times)
                remaining_chunks = num_chunks - (chunk_index + 1)
                estimated_remaining = remaining_chunks * avg_chunk_time
                remaining_min = int(estimated_remaining // 60)
                remaining_sec = int(estimated_remaining % 60)

                print(
                    f"Processing chunk {chunk_index + 1}/{num_chunks} ({progress:.1f}%) - "
                    f"{elapsed:.1f}s elapsed, ~{remaining_min}m {remaining_sec}s remaining...",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Processing chunk {chunk_index + 1}/{num_chunks} "
                    f"({progress:.1f}%) - {elapsed:.1f}s elapsed...",
                    file=sys.stderr,
                )

            # Create chunk
            debug_log(
                f"Creating chunk for time range "
                f"{start_time:.1f}s - {start_time + actual_duration:.1f}s"
            )
            chunk_path = create_video_chunk(video_path, start_time, actual_duration, chunk_index)
            chunk_paths.append(chunk_path)

            # Process chunk
            debug_log(f"Processing chunk {chunk_index} with OpenCV")
            chunk_detections = detect_image_in_video_direct(
                chunk_path, template, threshold, frame_skip, scales, use_single_scale, num_cores
            )

            # Adjust timestamps to global video time
            adjusted_detections = [t + start_time for t in chunk_detections]
            detection_times.extend(adjusted_detections)

            chunk_time = time.time() - chunk_start_time
            chunk_times.append(chunk_time)
            debug_log(f"Chunk {chunk_index + 1} completed in {chunk_time:.2f}s")

            if adjusted_detections:
                print(
                    f"  Found {len(adjusted_detections)} matches in chunk "
                    f"{chunk_index + 1} at times: "
                    f"{[f'{t:.1f}s' for t in adjusted_detections]}",
                    file=sys.stderr,
                )
            else:
                debug_log(f"No matches found in chunk {chunk_index + 1}")

    finally:
        # Clean up all chunk files
        debug_log(f"Cleaning up {len(chunk_paths)} temporary chunk files")
        for chunk_path in chunk_paths:
            cleanup_temp_file(chunk_path)

    total_time = time.time() - total_start_time

    # Display performance statistics
    realtime_factor = total_duration / total_time if total_time > 0 else 0
    avg_chunk_time = sum(chunk_times) / len(chunk_times) if chunk_times else 0

    print(
        f"Completed in {total_time / 60:.1f}m "
        f"({realtime_factor:.1f}x realtime, {avg_chunk_time:.1f}s/chunk). "
        f"Found {len(detection_times)} matches.",
        file=sys.stderr,
    )
    return detection_times


def detect_image_in_video_direct(
    video_path: Path,
    template: np.ndarray,
    threshold: float,
    frame_skip: int = 1,
    scales: list[float] | None = None,
    use_single_scale: bool = False,
    num_cores: int = 0,
) -> list[float]:
    debug_log(f"Starting direct video processing for {video_path.name}")
    temp_video_path: Path | None = None

    try:
        cap = None
        working_path = video_path

        # Try multiple OpenCV backends in order of preference
        backends = [
            (cv2.CAP_FFMPEG, "FFMPEG"),
            (cv2.CAP_GSTREAMER, "GStreamer"),
            (cv2.CAP_V4L2, "V4L2"),
            (cv2.CAP_ANY, "ANY"),
        ]

        for backend_id, backend_name in backends:
            debug_log(f"Trying {backend_name} backend")
            try:
                cap = cv2.VideoCapture(str(working_path), backend_id)
                if cap.isOpened():
                    # Test if we can actually read frames
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None and test_frame.size > 0:
                        # Success! Reset to beginning
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        debug_log(f"Successfully opened video with {backend_name} backend")
                        break
                    else:
                        debug_log(f"{backend_name} backend opened but cannot read frames")
                        cap.release()
                        cap = None
                else:
                    debug_log(f"{backend_name} backend failed to open video")
                    if cap:
                        cap.release()
                    cap = None
            except Exception as e:
                debug_log(f"{backend_name} backend threw exception: {e}")
                if cap:
                    cap.release()
                cap = None
                continue

        # If all backends failed, try conversion
        if cap is None or not cap.isOpened():
            debug_log("All backends failed, attempting video conversion to H.264")
            print(
                "Video format not supported by OpenCV backends, attempting conversion...",
                file=sys.stderr,
            )
            temp_video_path = convert_video_to_h264(video_path)
            if temp_video_path is None:
                raise ValueError("Video conversion failed")
            working_path = temp_video_path

            # Try backends again with converted video
            for backend_id, backend_name in backends:
                debug_log(f"Trying {backend_name} backend with converted video")
                try:
                    cap = cv2.VideoCapture(str(working_path), backend_id)
                    if cap.isOpened():
                        ret, test_frame = cap.read()
                        if ret and test_frame is not None and test_frame.size > 0:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            debug_log(
                                f"Successfully opened converted video with {backend_name} backend"
                            )
                            break
                        else:
                            cap.release()
                            cap = None
                    else:
                        if cap:
                            cap.release()
                        cap = None
                except Exception as e:
                    debug_log(f"{backend_name} backend failed with converted video: {e}")
                    if cap:
                        cap.release()
                    cap = None
                    continue

        if cap is None or not cap.isOpened():
            raise ValueError(f"Could not open video with any backend: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30.0  # Fallback FPS if cannot be determined

        debug_log(
            f"Video properties: FPS={fps:.2f}, Total frames={total_frames}, "
            f"Duration={total_frames / fps:.1f}s, Frame skip={frame_skip}"
        )

        # Determine number of cores to use
        if num_cores <= 0:
            num_cores = multiprocessing.cpu_count()

        debug_log(f"Using {num_cores} cores for processing")

        # Display performance information
        props = get_video_properties(working_path)
        duration_min = (total_frames / fps) / 60
        scale_mode = "single-scale" if use_single_scale else "multi-scale"

        print(
            f"Processing {duration_min:.1f} minute video "
            f"({props['width']}x{props['height']}, {props['fps']}fps, {props['codec']})",
            file=sys.stderr,
        )
        print(
            f"Using {num_cores} cores, {scale_mode} matching, "
            f"processing every {frame_skip} frame(s)",
            file=sys.stderr,
        )

        # Suggest optimizations if not already enabled
        if frame_skip == 1 and not use_single_scale:
            print(
                "Tip: Use --frame-skip 5 for ~5x faster processing "
                "or --single-scale for ~5x faster matching",
                file=sys.stderr,
            )

        detection_times: list[float] = []
        frame_position = 0  # Actual position in video
        processed_frames = 0
        currently_detected = False

        last_progress_time = time.time()
        process_start_time = time.time()

        # Collect frames for batch processing
        frame_batch = []
        batch_size = num_cores * 4  # Process frames in batches

        while True:
            # Check for interruption
            if INTERRUPTED:
                debug_log(f"Processing interrupted at frame {frame_position}")
                break

            ret, frame = cap.read()
            if not ret:
                break

            # Skip invalid frames
            if frame is None or frame.size == 0:
                frame_position += 1
                continue

            # Frame skipping logic
            if frame_position % frame_skip != 0:
                frame_position += 1
                continue

            # Add frame to batch
            frame_batch.append((frame_position, frame.copy()))
            processed_frames += 1
            frame_position += 1

            # Process batch when it's full or at end of video
            if len(frame_batch) >= batch_size:
                # Check for interruption
                if INTERRUPTED:
                    debug_log(f"Processing interrupted. Processed {processed_frames} frames.")
                    if detection_times:
                        print(
                            f"Processed {processed_frames}/{total_frames} frames "
                            f"before interruption. Found {len(detection_times)} matches.",
                            file=sys.stderr,
                        )
                    break

                # Process this batch
                batch_results = process_frame_batch(
                    frame_batch, template, threshold, scales, use_single_scale, fps, num_cores
                )

                # Apply state tracking to results
                for frame_idx, timestamp, is_match, confidence, best_scale in batch_results:
                    if is_match and not currently_detected:
                        detection_times.append(timestamp)
                        currently_detected = True
                        if use_single_scale:
                            debug_log(
                                f"Match found at frame {frame_idx}, timestamp {timestamp:.2f}s"
                            )
                        else:
                            debug_log(
                                f"Match found at frame {frame_idx}, timestamp {timestamp:.2f}s, "
                                f"confidence {confidence:.4f}, scale {best_scale:.2f}x"
                            )
                    elif not is_match and currently_detected:
                        currently_detected = False

                frame_batch = []

            # Progress update every 5 seconds for debug mode or every 30 seconds otherwise
            current_time = time.time()
            progress_interval = 5 if DEBUG else 30
            if current_time - last_progress_time > progress_interval and total_frames > 0:
                progress = frame_position / total_frames * 100
                elapsed = current_time - process_start_time

                # Calculate time remaining estimate
                if processed_frames > 100:  # Wait for some frames to get accurate estimate
                    frames_per_second = processed_frames / elapsed
                    remaining_frames = total_frames - frame_position
                    estimated_remaining = remaining_frames / frames_per_second
                    remaining_min = int(estimated_remaining // 60)
                    remaining_sec = int(estimated_remaining % 60)

                    if DEBUG:
                        debug_log(
                            f"Frame {frame_position}/{total_frames} ({progress:.1f}%), "
                            f"processed {processed_frames} frames, {elapsed:.1f}s elapsed, "
                            f"~{remaining_min}m {remaining_sec}s remaining"
                        )
                    else:
                        print(
                            f"Frame {frame_position}/{total_frames} ({progress:.1f}%) - "
                            f"{elapsed:.1f}s elapsed, "
                            f"~{remaining_min}m {remaining_sec}s remaining...",
                            file=sys.stderr,
                        )
                else:
                    if DEBUG:
                        debug_log(
                            f"Frame {frame_position}/{total_frames} ({progress:.1f}%), "
                            f"processed {processed_frames} frames, {elapsed:.1f}s elapsed"
                        )
                    else:
                        print(
                            f"Frame {frame_position}/{total_frames} "
                            f"({progress:.1f}%) - {elapsed:.1f}s elapsed...",
                            file=sys.stderr,
                        )

                last_progress_time = current_time

        # Process remaining frames in the final batch
        if frame_batch:
            batch_results = process_frame_batch(
                frame_batch, template, threshold, scales, use_single_scale, fps, num_cores
            )

            # Apply state tracking to results
            for frame_idx, timestamp, is_match, confidence, best_scale in batch_results:
                if is_match and not currently_detected:
                    detection_times.append(timestamp)
                    currently_detected = True
                    if use_single_scale:
                        debug_log(f"Match found at frame {frame_idx}, timestamp {timestamp:.2f}s")
                    else:
                        debug_log(
                            f"Match found at frame {frame_idx}, timestamp {timestamp:.2f}s, "
                            f"confidence {confidence:.4f}, scale {best_scale:.2f}x"
                        )
                elif not is_match and currently_detected:
                    currently_detected = False

        cap.release()
        process_time = time.time() - process_start_time

        # Display performance statistics
        video_duration = total_frames / fps
        realtime_factor = video_duration / process_time if process_time > 0 else 0
        frames_per_sec = processed_frames / process_time if process_time > 0 else 0

        print(
            f"Completed in {process_time / 60:.1f}m "
            f"({realtime_factor:.1f}x realtime, {frames_per_sec:.1f} frames/s). "
            f"Found {len(detection_times)} matches.",
            file=sys.stderr,
        )

        debug_log(
            f"Direct processing completed in {process_time:.2f}s, processed {frame_position} frames"
        )
        return detection_times

    finally:
        cleanup_temp_file(temp_video_path)


def detect_image_in_video_fast(
    video_path: Path,
    template: np.ndarray,
    threshold: float,
    fps: float = 2.0,
    max_height: int = 720,
    scales: list[float] | None = None,
    use_single_scale: bool = False,
    num_cores: int = 0,
    use_grayscale: bool = True,
) -> list[float]:
    """
    Fast video processing using ffmpeg frame extraction.
    Pre-extracts frames at low FPS/resolution, then processes in parallel.
    """
    debug_log(f"Starting fast mode processing for {video_path.name}")
    debug_log(f"Settings: fps={fps}, max_height={max_height}, grayscale={use_grayscale}")

    duration = get_video_duration(video_path)
    props = get_video_properties(video_path)

    print(
        f"Fast mode: Processing {duration / 60:.1f} minute video "
        f"({props['width']}x{props['height']}, {props['codec']})",
        file=sys.stderr,
    )
    print(
        f"Extracting frames at {fps} fps, max height {max_height}px, "
        f"{'grayscale' if use_grayscale else 'color'} matching",
        file=sys.stderr,
    )

    frames_dir = None
    try:
        # Extract frames using ffmpeg
        print("Extracting frames...", file=sys.stderr)
        frames_dir, num_frames = extract_frames_to_dir(video_path, fps, max_height)

        video_duration = num_frames / fps
        print(
            f"Extracted {num_frames} frames "
            f"({video_duration / 60:.1f} minutes of video at {fps} fps)",
            file=sys.stderr,
        )

        # Process extracted frames
        print("Analyzing frames...", file=sys.stderr)
        detection_times = process_extracted_frames(
            frames_dir,
            template,
            threshold,
            fps,
            scales,
            use_single_scale,
            num_cores,
            use_grayscale,
        )

        return detection_times

    finally:
        cleanup_temp_dir(frames_dir)


def detect_image_in_video(
    video_path: Path,
    template: np.ndarray,
    threshold: float,
    chunk_size: int = 60,
    frame_skip: int = 1,
    scales: list[float] | None = None,
    use_single_scale: bool = False,
    num_cores: int = 0,
    force_chunking: bool = False,
    fast_mode: bool = False,
    fps: float = 2.0,
    max_height: int = 720,
    use_grayscale: bool = True,
) -> list[float]:
    """Main entry point for video processing with automatic fallback to chunking if needed."""
    debug_log(f"Starting video processing for {video_path.name}")
    duration = get_video_duration(video_path)
    debug_log(f"Video duration: {duration:.1f}s, chunk size: {chunk_size}s")

    # Use fast mode if enabled (default)
    if fast_mode:
        debug_log("Using fast mode with ffmpeg frame extraction")
        return detect_image_in_video_fast(
            video_path,
            template,
            threshold,
            fps,
            max_height,
            scales,
            use_single_scale,
            num_cores,
            use_grayscale,
        )

    if force_chunking:
        debug_log("Force chunking enabled, using chunked processing")
        return process_video_in_chunks(
            video_path,
            template,
            threshold,
            chunk_size,
            frame_skip,
            scales,
            use_single_scale,
            num_cores,
        )

    # Try direct processing (traditional frame-by-frame via OpenCV)
    debug_log(f"Video is {duration / 60:.1f} minutes, attempting direct processing")
    try:
        return detect_image_in_video_direct(
            video_path, template, threshold, frame_skip, scales, use_single_scale, num_cores
        )
    except Exception as e:
        debug_log(f"Direct processing failed: {e}")
        print(f"Direct processing failed: {e}", file=sys.stderr)
        if duration > 0:
            print("Attempting chunked processing as fallback...", file=sys.stderr)
            return process_video_in_chunks(
                video_path,
                template,
                threshold,
                chunk_size,
                frame_skip,
                scales,
                use_single_scale,
                num_cores,
            )
        else:
            raise


def main() -> None:
    global DEBUG

    # Register signal handler for graceful interruption
    signal.signal(signal.SIGINT, signal_handler)

    try:
        args = parse_arguments()
        DEBUG = args.debug

        if DEBUG:
            print("Debug mode enabled", file=sys.stderr)
            debug_log(
                f"Arguments: video={args.video_path}, image={args.image_path}, "
                f"threshold={args.threshold}, chunk_size={args.chunk_size}, "
                f"scales={args.scales}, single_scale={args.single_scale}, cores={args.cores}, "
                f"fast_mode={args.fast_mode}, fps={args.fps}, max_height={args.max_height}, "
                f"grayscale={args.grayscale}"
            )

        if not args.video_path.exists():
            print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
            sys.exit(1)

        if not args.image_path.exists():
            print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
            sys.exit(1)

        # Load template with grayscale flag if fast mode is enabled
        use_grayscale = args.grayscale and args.fast_mode
        debug_log(f"Loading template image (grayscale={use_grayscale})")
        template = load_template(args.image_path, grayscale=use_grayscale)
        debug_log(f"Template loaded: {template.shape}")

        # Parse scales
        try:
            scales = parse_scales(args.scales)
            debug_log(f"Using scales: {scales}")
        except ValueError as e:
            print(f"Error parsing scales '{args.scales}': {e}", file=sys.stderr)
            sys.exit(1)

        use_single_scale = args.single_scale
        if use_single_scale:
            debug_log("Using single-scale matching mode")
        else:
            debug_log("Using multi-scale matching mode")

        debug_log("Starting video processing")
        detection_times = detect_image_in_video(
            args.video_path,
            template,
            args.threshold,
            args.chunk_size,
            args.frame_skip,
            scales,
            use_single_scale,
            args.cores,
            args.force_chunking,
            args.fast_mode,
            args.fps,
            args.max_height,
            use_grayscale,
        )

        debug_log(f"Processing complete, found {len(detection_times)} matches")
        if detection_times:
            for time in detection_times:
                minutes = int(time // 60)
                seconds = time % 60
                print(f"{minutes:02d}:{seconds:06.3f}")
        else:
            print("No matches found")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        if not INTERRUPTED:
            print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
