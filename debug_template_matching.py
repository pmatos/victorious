#!/usr/bin/env python3
"""
Debug script for template matching issues in video processing.
Extracts frames and performs comprehensive template matching analysis.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def extract_frames_with_ffmpeg(
    video_path: Path, output_dir: Path, frame_skip: int = 60
) -> list[Path]:
    """Extract every Nth frame from video using ffmpeg."""
    print(f"Extracting every {frame_skip}th frame from {video_path.name}...")

    # Get video properties first
    probe_cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration,stream=r_frame_rate",
        "-of",
        "csv=p=0",
        str(video_path),
    ]

    try:
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
        lines = result.stdout.strip().split("\n")
        duration = float(lines[0]) if lines[0] else 0
        fps_str = lines[1] if len(lines) > 1 else "30/1"

        # Parse framerate (e.g., "60/1" -> 60.0)
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired, IndexError):
        duration = 60  # fallback
        fps = 30.0  # fallback

    print(f"Video: {duration:.1f}s duration, {fps:.1f} FPS")

    # Extract frames using ffmpeg with frame skipping
    # Use fps filter to extract every Nth frame
    filter_fps = fps / frame_skip

    cmd = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vf",
        f"fps={filter_fps:.3f}",
        "-f",
        "image2",
        str(output_dir / "frame_%06d.png"),
    ]

    try:
        print(f"Running: {' '.join(cmd[:6])}...")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            return []

    except subprocess.TimeoutExpired:
        print("Frame extraction timed out")
        return []

    # Get list of extracted frames
    frame_files = sorted(output_dir.glob("frame_*.png"))
    print(f"Extracted {len(frame_files)} frames")

    return frame_files


def load_template_image(image_path: Path) -> np.ndarray:
    """Load template image."""
    template = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if template is None:
        raise ValueError(f"Could not load template image: {image_path}")
    return template


def multi_scale_template_matching(
    frame: np.ndarray,
    template: np.ndarray,
    scales: list[float] | None = None,
    methods: list[tuple[int, str]] | None = None,
) -> dict[str, Any]:
    """
    Perform template matching at multiple scales and with different methods.
    Returns detailed results for analysis.
    """
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    if methods is None:
        methods = [
            (cv2.TM_CCOEFF_NORMED, "TM_CCOEFF_NORMED"),
            (cv2.TM_CCORR_NORMED, "TM_CCORR_NORMED"),
            (cv2.TM_SQDIFF_NORMED, "TM_SQDIFF_NORMED"),
        ]

    results: dict[str, Any] = {
        "best_match": None,
        "best_confidence": 0.0,
        "best_location": None,
        "best_scale": 1.0,
        "best_method": "",
        "all_results": [],
    }

    template_h, template_w = template.shape[:2]

    for scale in scales:
        # Resize template
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)

        if new_w > frame.shape[1] or new_h > frame.shape[0]:
            continue  # Template too large for frame

        if new_w < 10 or new_h < 10:
            continue  # Template too small

        scaled_template = cv2.resize(template, (new_w, new_h))

        for method_id, method_name in methods:
            try:
                # Perform template matching
                result = cv2.matchTemplate(frame, scaled_template, method_id)

                # Get best match location and confidence
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

                # For TM_SQDIFF_NORMED, lower values are better matches
                if method_id == cv2.TM_SQDIFF_NORMED:
                    confidence = 1.0 - min_val
                    match_loc = min_loc
                else:
                    confidence = max_val
                    match_loc = max_loc

                match_result = {
                    "scale": scale,
                    "method": method_name,
                    "confidence": confidence,
                    "location": match_loc,
                    "template_size": (new_w, new_h),
                    "raw_min_val": min_val,
                    "raw_max_val": max_val,
                }

                results["all_results"].append(match_result)

                # Update best match
                if confidence > results["best_confidence"]:
                    results["best_match"] = match_result
                    results["best_confidence"] = confidence
                    results["best_location"] = match_loc
                    results["best_scale"] = scale
                    results["best_method"] = method_name

            except cv2.error as e:
                print(f"OpenCV error with scale {scale}, method {method_name}: {e}")
                continue

    return results


def draw_match_visualization(
    frame: np.ndarray, template: np.ndarray, match_result: dict[str, Any], output_path: Path
) -> None:
    """Draw bounding box around best match and save visualization."""

    if match_result["best_match"] is None:
        # No match found, just save the original frame
        cv2.imwrite(str(output_path), frame)
        return

    # Create a copy for drawing
    vis_frame = frame.copy()

    best_match = match_result["best_match"]
    location = best_match["location"]
    template_size = best_match["template_size"]
    confidence = best_match["confidence"]
    scale = best_match["scale"]
    method = best_match["method"]

    # Draw bounding box
    top_left = location
    bottom_right = (location[0] + template_size[0], location[1] + template_size[1])

    # Color based on confidence: red for low, green for high
    if confidence >= 0.8:
        color = (0, 255, 0)  # Green - high confidence
    elif confidence >= 0.6:
        color = (0, 255, 255)  # Yellow - medium confidence
    else:
        color = (0, 0, 255)  # Red - low confidence

    cv2.rectangle(vis_frame, top_left, bottom_right, color, 2)

    # Add text with match info
    text = f"{confidence:.3f} @{scale:.2f}x {method}"
    cv2.putText(
        vis_frame, text, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
    )

    # Save visualization
    cv2.imwrite(str(output_path), vis_frame)


def analyze_frames(
    frame_files: list[Path], template: np.ndarray, output_dir: Path, min_confidence: float = 0.5
) -> list[dict[str, Any]]:
    """Analyze all extracted frames for template matches."""

    results = []
    template_h, template_w = template.shape[:2]

    print(f"Analyzing {len(frame_files)} frames...")
    print(f"Template size: {template_w}x{template_h}")

    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        if frame is None:
            print(f"Could not load frame: {frame_file}")
            continue

        # Extract frame number from filename for timestamp calculation
        frame_num_str = frame_file.stem.split("_")[-1]
        try:
            frame_index = int(frame_num_str)
        except ValueError:
            frame_index = i

        # Perform multi-scale template matching
        match_result = multi_scale_template_matching(frame, template)

        # Add frame info
        match_result["frame_file"] = frame_file
        match_result["frame_index"] = frame_index
        match_result["frame_size"] = frame.shape[:2]

        results.append(match_result)

        # Create visualization
        vis_path = output_dir / f"vis_{frame_file.name}"
        draw_match_visualization(frame, template, match_result, vis_path)

        # Print progress and significant matches
        if match_result["best_confidence"] >= min_confidence:
            best = match_result["best_match"]
            print(
                f"Frame {i + 1:3d}: MATCH! Confidence: {match_result['best_confidence']:.3f}, "
                f"Scale: {best['scale']:.2f}x, Method: {best['method']}, "
                f"Location: {best['location']}"
            )
        elif i % 10 == 0:
            print(f"Frame {i + 1:3d}: Best confidence: {match_result['best_confidence']:.3f}")

    return results


def generate_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Generate detailed analysis report."""

    with output_path.open("w") as f:
        f.write("Template Matching Analysis Report\n")
        f.write("==================================\n\n")

        # Summary statistics
        total_frames = len(results)
        matches_50 = sum(1 for r in results if r["best_confidence"] >= 0.5)
        matches_60 = sum(1 for r in results if r["best_confidence"] >= 0.6)
        matches_70 = sum(1 for r in results if r["best_confidence"] >= 0.7)
        matches_80 = sum(1 for r in results if r["best_confidence"] >= 0.8)

        f.write(f"Total frames analyzed: {total_frames}\n")
        f.write(f"Matches >= 0.5 confidence: {matches_50}\n")
        f.write(f"Matches >= 0.6 confidence: {matches_60}\n")
        f.write(f"Matches >= 0.7 confidence: {matches_70}\n")
        f.write(f"Matches >= 0.8 confidence: {matches_80}\n\n")

        # Best matches
        sorted_results = sorted(results, key=lambda x: x["best_confidence"], reverse=True)

        f.write("Top 10 Matches:\n")
        f.write("---------------\n")
        for i, result in enumerate(sorted_results[:10]):
            best = result["best_match"]
            if best:
                f.write(
                    f"{i + 1:2d}. Frame: {result['frame_file'].name}, "
                    f"Confidence: {result['best_confidence']:.4f}, "
                    f"Scale: {best['scale']:.2f}x, "
                    f"Method: {best['method']}, "
                    f"Location: {best['location']}\n"
                )

        f.write("\nDetailed Results:\n")
        f.write("-----------------\n")

        for result in sorted_results:
            if result["best_confidence"] >= 0.5:  # Only report significant matches
                best = result["best_match"]
                f.write(f"\nFrame: {result['frame_file'].name}\n")
                f.write(f"  Frame size: {result['frame_size']}\n")
                f.write(f"  Best match: {result['best_confidence']:.4f} confidence\n")
                if best:
                    f.write(f"  Best scale: {best['scale']:.2f}x\n")
                    f.write(f"  Best method: {best['method']}\n")
                    f.write(f"  Location: {best['location']}\n")
                    f.write(f"  Template size at scale: {best['template_size']}\n")

                # Show top 3 results for this frame
                frame_results = sorted(
                    result["all_results"], key=lambda x: x["confidence"], reverse=True
                )[:3]
                f.write("  Top matches:\n")
                for j, match in enumerate(frame_results):
                    f.write(
                        f"    {j + 1}. {match['confidence']:.4f} @ {match['scale']:.2f}x "
                        f"{match['method']} {match['location']}\n"
                    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug template matching in video")
    parser.add_argument("video_path", type=Path, help="Path to video file")
    parser.add_argument("template_path", type=Path, help="Path to template image")
    parser.add_argument(
        "--frame-skip", type=int, default=60, help="Extract every Nth frame (default: 60)"
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence to report (default: 0.5)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default="debug_output", help="Output directory for results"
    )

    args = parser.parse_args()

    # Validate inputs
    if not args.video_path.exists():
        print(f"Error: Video file not found: {args.video_path}")
        sys.exit(1)

    if not args.template_path.exists():
        print(f"Error: Template image not found: {args.template_path}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(exist_ok=True)
    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    try:
        # Load template
        print(f"Loading template image: {args.template_path}")
        template = load_template_image(args.template_path)
        print(f"Template size: {template.shape[1]}x{template.shape[0]} pixels")

        # Extract frames
        frame_files = extract_frames_with_ffmpeg(args.video_path, frames_dir, args.frame_skip)

        if not frame_files:
            print("No frames extracted. Exiting.")
            sys.exit(1)

        # Analyze frames
        results = analyze_frames(frame_files, template, args.output_dir, args.min_confidence)

        # Generate report
        report_path = args.output_dir / "analysis_report.txt"
        generate_report(results, report_path)

        print("\nAnalysis complete!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Report: {report_path}")
        print(f"Visualizations: {args.output_dir}/vis_*.png")

        # Print summary
        matches = sum(1 for r in results if r["best_confidence"] >= args.min_confidence)
        print(f"\nSummary: {matches} frames with confidence >= {args.min_confidence}")

        if matches > 0:
            best_result = max(results, key=lambda x: x["best_confidence"])
            print(
                f"Best match: {best_result['best_confidence']:.4f} confidence "
                f"in {best_result['frame_file'].name}"
            )

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
