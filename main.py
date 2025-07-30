import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

# Global debug flag
DEBUG = False


def debug_log(message: str) -> None:
    """Print debug message if debug mode is enabled."""
    if DEBUG:
        print(f"DEBUG: {message}", file=sys.stderr)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect image appearances in video frames")
    parser.add_argument("video_path", type=Path, help="Path to the video file")
    parser.add_argument("image_path", type=Path, help="Path to the image to detect")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Matching threshold (0.0-1.0, default: 0.8)"
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
    return parser.parse_args()


def load_template(image_path: Path) -> np.ndarray:
    template = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"Could not load image: {image_path}")
    return template


def match_template_in_frame(frame: np.ndarray, template: np.ndarray, threshold: float) -> bool:
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val >= threshold


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

        # For very long videos (>30 min), process only first 30 minutes to avoid timeout
        if duration > 1800:  # 30 minutes
            cmd.extend(["-t", "1800"])
            print(
                f"Video is {duration / 60:.1f} minutes long. Processing first 30 minutes only.",
                file=sys.stderr,
            )

        cmd.extend(["-y", temp_path])

        timeout = min(300, max(60, duration * 0.5)) if duration > 0 else 300
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=timeout)
        return Path(temp_path)

    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.unlink(temp_path)

        if isinstance(e, FileNotFoundError):
            raise ValueError(
                "FFmpeg not found. Please install FFmpeg to process this video format."
            )
        elif isinstance(e, subprocess.TimeoutExpired):
            raise ValueError("Video conversion timed out. The video may be too large or complex.")
        else:
            raise ValueError(f"Video conversion failed: {e.stderr}")


def cleanup_temp_file(temp_path: Path | None) -> None:
    """Clean up temporary file if it exists."""
    if temp_path and temp_path.exists():
        try:
            temp_path.unlink()
        except OSError:
            pass


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


def create_video_chunk(
    input_path: Path, start_time: float, duration: float, chunk_index: int
) -> Path:
    """Create a temporary video chunk for processing."""
    debug_log(f"Creating chunk {chunk_index} from {start_time:.1f}s for {duration:.1f}s")
    start_chunk_time = time.time()

    temp_fd, temp_path = tempfile.mkstemp(suffix=f"_chunk_{chunk_index}.mp4", prefix="victorious_")
    os.close(temp_fd)

    # Use stream copy when possible to avoid re-encoding
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
        "libx264",  # Still need to transcode for AV1 compatibility
        "-preset",
        "ultrafast",
        "-crf",
        "28",
        "-avoid_negative_ts",
        "make_zero",  # Handle timestamp issues
        "-f",
        "mp4",
        "-y",
        temp_path,
    ]

    try:
        debug_log(f"Running FFmpeg command: {' '.join(cmd[:8])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=120)

        chunk_time = time.time() - start_chunk_time
        debug_log(f"Chunk {chunk_index} created in {chunk_time:.2f}s")

        return Path(temp_path)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        debug_log(f"Chunk creation failed: {e}")
        raise ValueError(f"Failed to create video chunk: {e}")


def process_video_in_chunks(
    video_path: Path,
    template: np.ndarray,
    threshold: float,
    chunk_duration: float = 60.0,
    frame_skip: int = 1,
) -> list[float]:
    """Process video in chunks with progress updates."""
    debug_log(f"Starting chunked processing with {chunk_duration}s chunks")
    total_start_time = time.time()

    total_duration = get_video_duration(video_path)
    debug_log(f"Video duration: {total_duration:.1f}s ({total_duration / 60:.1f} minutes)")

    if total_duration <= 0:
        debug_log("Could not determine video duration, falling back to direct processing")
        return detect_image_in_video_direct(video_path, template, threshold, frame_skip)

    print(
        f"Processing {total_duration / 60:.1f} minute video in {chunk_duration:.0f}s chunks...",
        file=sys.stderr,
    )

    detection_times = []
    chunk_paths = []
    num_chunks = int((total_duration + chunk_duration - 1) // chunk_duration)  # Ceiling division
    debug_log(f"Will create {num_chunks} chunks")

    try:
        for chunk_index in range(num_chunks):
            chunk_start_time = time.time()
            start_time = chunk_index * chunk_duration
            remaining_time = total_duration - start_time
            actual_duration = min(chunk_duration, remaining_time)

            progress = (chunk_index + 1) / num_chunks * 100
            elapsed = time.time() - total_start_time
            print(
                f"Processing chunk {chunk_index + 1}/{num_chunks} ({progress:.1f}%) - {elapsed:.1f}s elapsed...",
                file=sys.stderr,
            )

            # Create chunk
            debug_log(
                f"Creating chunk for time range {start_time:.1f}s - {start_time + actual_duration:.1f}s"
            )
            chunk_path = create_video_chunk(video_path, start_time, actual_duration, chunk_index)
            chunk_paths.append(chunk_path)

            # Process chunk
            debug_log(f"Processing chunk {chunk_index} with OpenCV")
            chunk_detections = detect_image_in_video_direct(
                chunk_path, template, threshold, frame_skip
            )

            # Adjust timestamps to global video time
            adjusted_detections = [t + start_time for t in chunk_detections]
            detection_times.extend(adjusted_detections)

            chunk_time = time.time() - chunk_start_time
            debug_log(f"Chunk {chunk_index + 1} completed in {chunk_time:.2f}s")

            if adjusted_detections:
                print(
                    f"  Found {len(adjusted_detections)} matches in chunk {chunk_index + 1} at times: {[f'{t:.1f}s' for t in adjusted_detections]}",
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
    print(
        f"Completed processing in {total_time:.1f}s. Found {len(detection_times)} total matches.",
        file=sys.stderr,
    )
    return detection_times


def detect_image_in_video_direct(
    video_path: Path, template: np.ndarray, threshold: float, frame_skip: int = 1
) -> list[float]:
    debug_log(f"Starting direct video processing for {video_path.name}")
    temp_video_path = None

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
            f"Video properties: FPS={fps:.2f}, Total frames={total_frames}, Duration={total_frames / fps:.1f}s, Frame skip={frame_skip}"
        )

        detection_times: list[float] = []
        frame_count = 0
        processed_frames = 0
        currently_detected = False

        last_progress_time = time.time()
        process_start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip invalid frames
            if frame is None or frame.size == 0:
                frame_count += 1
                continue

            # Frame skipping logic
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue

            processed_frames += 1
            timestamp = frame_count / fps
            is_match = match_template_in_frame(frame, template, threshold)

            if is_match and not currently_detected:
                detection_times.append(timestamp)
                currently_detected = True
                debug_log(f"Match found at frame {frame_count}, timestamp {timestamp:.2f}s")
            elif not is_match and currently_detected:
                currently_detected = False

            frame_count += 1

            # Progress update every 5 seconds for debug mode or every 30 seconds otherwise
            current_time = time.time()
            progress_interval = 5 if DEBUG else 30
            if current_time - last_progress_time > progress_interval and total_frames > 0:
                progress = frame_count / total_frames * 100
                elapsed = current_time - process_start_time
                debug_log(
                    f"Frame {frame_count}/{total_frames} ({progress:.1f}%), processed {processed_frames} frames, {elapsed:.1f}s elapsed"
                )
                last_progress_time = current_time

        cap.release()
        process_time = time.time() - process_start_time
        debug_log(
            f"Direct processing completed in {process_time:.2f}s, processed {frame_count} frames"
        )
        return detection_times

    finally:
        cleanup_temp_file(temp_video_path)


def detect_image_in_video(
    video_path: Path,
    template: np.ndarray,
    threshold: float,
    chunk_size: int = 60,
    frame_skip: int = 1,
) -> list[float]:
    """Main entry point for video processing with automatic chunking for large videos."""
    debug_log(f"Starting video processing for {video_path.name}")
    duration = get_video_duration(video_path)
    debug_log(f"Video duration: {duration:.1f}s, chunk size: {chunk_size}s")

    # Use chunked processing for videos longer than 5 minutes or if direct processing fails
    if duration > 300:  # 5 minutes
        debug_log(f"Video is {duration / 60:.1f} minutes, using chunked processing")
        try:
            return process_video_in_chunks(video_path, template, threshold, chunk_size, frame_skip)
        except Exception as e:
            debug_log(f"Chunked processing failed: {e}")
            print(f"Chunked processing failed: {e}", file=sys.stderr)
            print("Falling back to direct processing...", file=sys.stderr)
            return detect_image_in_video_direct(video_path, template, threshold, frame_skip)
    else:
        debug_log(f"Video is {duration / 60:.1f} minutes, using direct processing")
        try:
            return detect_image_in_video_direct(video_path, template, threshold, frame_skip)
        except Exception as e:
            debug_log(f"Direct processing failed: {e}")
            print(f"Direct processing failed: {e}", file=sys.stderr)
            if duration > 0:
                print("Attempting chunked processing as fallback...", file=sys.stderr)
                return process_video_in_chunks(
                    video_path, template, threshold, chunk_size, frame_skip
                )
            else:
                raise


def main() -> None:
    global DEBUG

    try:
        args = parse_arguments()
        DEBUG = args.debug

        if DEBUG:
            print("Debug mode enabled", file=sys.stderr)
            debug_log(
                f"Arguments: video={args.video_path}, image={args.image_path}, threshold={args.threshold}, chunk_size={args.chunk_size}"
            )

        if not args.video_path.exists():
            print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
            sys.exit(1)

        if not args.image_path.exists():
            print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
            sys.exit(1)

        debug_log("Loading template image")
        template = load_template(args.image_path)
        debug_log(f"Template loaded: {template.shape}")

        debug_log("Starting video processing")
        detection_times = detect_image_in_video(
            args.video_path, template, args.threshold, args.chunk_size, args.frame_skip
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
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if DEBUG:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
