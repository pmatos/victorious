import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect image appearances in video frames")
    parser.add_argument("video_path", type=Path, help="Path to the video file")
    parser.add_argument("image_path", type=Path, help="Path to the image to detect")
    parser.add_argument(
        "--threshold", type=float, default=0.8, help="Matching threshold (0.0-1.0, default: 0.8)"
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


def detect_image_in_video(video_path: Path, template: np.ndarray, threshold: float) -> list[float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    detection_times: list[float] = []
    frame_count = 0
    currently_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_count / fps
        is_match = match_template_in_frame(frame, template, threshold)

        if is_match and not currently_detected:
            detection_times.append(timestamp)
            currently_detected = True
        elif not is_match and currently_detected:
            currently_detected = False

        frame_count += 1

    cap.release()
    return detection_times


def main() -> None:
    try:
        args = parse_arguments()

        if not args.video_path.exists():
            print(f"Error: Video file not found: {args.video_path}", file=sys.stderr)
            sys.exit(1)

        if not args.image_path.exists():
            print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
            sys.exit(1)

        template = load_template(args.image_path)
        detection_times = detect_image_in_video(args.video_path, template, args.threshold)

        if detection_times:
            for time in detection_times:
                print(f"{time:.2f}")
        else:
            print("No matches found")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
