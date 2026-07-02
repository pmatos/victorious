#!/usr/bin/env python3
"""
Improved analysis to identify real matches vs false positives.
Uses TM_CCOEFF_NORMED (the method used in main.py) and applies stricter criteria.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def analyze_with_ccoeff_normed(
    frame_files: list[Path],
    template: np.ndarray,
    output_dir: Path,
    scales: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Analyze frames using TM_CCOEFF_NORMED method like main.py."""
    if scales is None:
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]

    results = []
    template_h, template_w = template.shape[:2]

    print(f"Analyzing {len(frame_files)} frames with TM_CCOEFF_NORMED...")
    print(f"Template size: {template_w}x{template_h}")

    for i, frame_file in enumerate(frame_files):
        frame = cv2.imread(str(frame_file), cv2.IMREAD_COLOR)
        if frame is None:
            continue

        frame_results = []
        best_confidence = 0.0
        best_match: dict[str, Any] | None = None

        # Test multiple scales
        for scale in scales:
            new_w = int(template_w * scale)
            new_h = int(template_h * scale)

            if new_w > frame.shape[1] or new_h > frame.shape[0] or new_w < 10 or new_h < 10:
                continue

            scaled_template = cv2.resize(template, (new_w, new_h))

            # Use TM_CCOEFF_NORMED like main.py
            result = cv2.matchTemplate(frame, scaled_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            match_data = {
                "scale": scale,
                "confidence": max_val,
                "location": max_loc,
                "template_size": (new_w, new_h),
            }

            frame_results.append(match_data)

            if max_val > best_confidence:
                best_confidence = max_val
                best_match = match_data

        # Store frame analysis
        frame_analysis = {
            "frame_file": frame_file,
            "frame_index": i,
            "best_confidence": best_confidence,
            "best_match": best_match,
            "all_matches": frame_results,
            "frame_size": frame.shape[:2],
        }

        results.append(frame_analysis)

        # Print significant matches only
        if best_confidence >= 0.7 and best_match is not None:
            print(
                f"Frame {i + 1:3d}: Confidence: {best_confidence:.4f}, "
                f"Scale: {best_match['scale']:.2f}x, "
                f"Location: {best_match['location']}"
            )

    return results


def create_enhanced_visualization(
    frame: np.ndarray,
    template: np.ndarray,
    match_data: dict[str, Any],
    output_path: Path,
    confidence_threshold: float = 0.7,
) -> None:
    """Create enhanced visualization showing multiple potential matches."""

    vis_frame = frame.copy()

    if not match_data["best_match"] or match_data["best_confidence"] < confidence_threshold:
        # Low confidence - just save original with text
        cv2.putText(
            vis_frame,
            f"Low confidence: {match_data['best_confidence']:.3f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imwrite(str(output_path), vis_frame)
        return

    # Sort matches by confidence
    sorted_matches = sorted(match_data["all_matches"], key=lambda x: x["confidence"], reverse=True)

    colors = [(0, 255, 0), (0, 255, 255), (255, 0, 255), (255, 0, 0), (0, 0, 255)]

    # Draw top 3 matches
    for i, match in enumerate(sorted_matches[:3]):
        if match["confidence"] < 0.5:  # Skip very low confidence
            continue

        location = match["location"]
        template_size = match["template_size"]
        confidence = match["confidence"]
        scale = match["scale"]

        # Draw bounding box
        top_left = location
        bottom_right = (location[0] + template_size[0], location[1] + template_size[1])

        color = colors[i % len(colors)]
        thickness = 3 if i == 0 else 2  # Thicker for best match

        cv2.rectangle(vis_frame, top_left, bottom_right, color, thickness)

        # Add text
        text = f"#{i + 1}: {confidence:.3f} @{scale:.1f}x"
        text_y = top_left[1] - 10 if top_left[1] > 20 else bottom_right[1] + 20
        cv2.putText(vis_frame, text, (top_left[0], text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite(str(output_path), vis_frame)


def identify_real_matches(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Identify likely real matches using statistical analysis."""

    # Extract all confidences
    all_confidences = [r["best_confidence"] for r in results]
    mean_confidence = np.mean(all_confidences)
    std_confidence = np.std(all_confidences)

    print("\nStatistical Analysis:")
    print(f"Mean confidence: {mean_confidence:.4f}")
    print(f"Std deviation: {std_confidence:.4f}")
    print(f"Min confidence: {min(all_confidences):.4f}")
    print(f"Max confidence: {max(all_confidences):.4f}")

    # Adaptive threshold: mean + 1.5 * std
    adaptive_threshold = mean_confidence + 1.5 * std_confidence
    print(f"Adaptive threshold: {adaptive_threshold:.4f}")

    # Find outliers (potential real matches)
    high_confidence_matches = []
    for result in results:
        if result["best_confidence"] > adaptive_threshold:
            high_confidence_matches.append(result)

    # Group nearby matches (same location across consecutive frames)
    grouped_matches = []
    if high_confidence_matches:
        # Sort by frame index
        high_confidence_matches.sort(key=lambda x: x["frame_index"])

        current_group = [high_confidence_matches[0]]

        for i in range(1, len(high_confidence_matches)):
            current = high_confidence_matches[i]
            prev = current_group[-1]

            # Check if locations are similar and frames are consecutive-ish
            prev_loc = prev["best_match"]["location"]
            curr_loc = current["best_match"]["location"]

            distance = np.sqrt((prev_loc[0] - curr_loc[0]) ** 2 + (prev_loc[1] - curr_loc[1]) ** 2)
            frame_gap = current["frame_index"] - prev["frame_index"]

            if distance < 100 and frame_gap <= 3:  # Same general area, close frames
                current_group.append(current)
            else:
                if len(current_group) >= 2:  # Group has multiple frames
                    grouped_matches.append(current_group)
                current_group = [current]

        # Don't forget the last group
        if len(current_group) >= 2:
            grouped_matches.append(current_group)

    return {
        "mean_confidence": mean_confidence,
        "std_confidence": std_confidence,
        "adaptive_threshold": adaptive_threshold,
        "high_confidence_matches": high_confidence_matches,
        "grouped_matches": grouped_matches,
        "all_confidences": all_confidences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze real matches vs false positives")
    parser.add_argument("template_path", type=Path, help="Path to template image")
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default="debug_output/frames",
        help="Directory with extracted frames",
    )
    parser.add_argument("--output-dir", type=Path, default="debug_output", help="Output directory")

    args = parser.parse_args()

    if not args.template_path.exists():
        print(f"Error: Template not found: {args.template_path}")
        sys.exit(1)

    if not args.frames_dir.exists():
        print(f"Error: Frames directory not found: {args.frames_dir}")
        sys.exit(1)

    # Load template
    template = cv2.imread(str(args.template_path), cv2.IMREAD_COLOR)
    if template is None:
        print(f"Error: Could not load template: {args.template_path}")
        sys.exit(1)

    # Get frame files
    frame_files = sorted(args.frames_dir.glob("frame_*.png"))
    if not frame_files:
        print(f"Error: No frame files found in {args.frames_dir}")
        sys.exit(1)

    print(f"Found {len(frame_files)} frames to analyze")

    # Analyze with TM_CCOEFF_NORMED
    results = analyze_with_ccoeff_normed(frame_files, template, args.output_dir)

    # Create enhanced visualizations
    print("\nCreating enhanced visualizations...")
    for result in results:
        frame = cv2.imread(str(result["frame_file"]), cv2.IMREAD_COLOR)
        if frame is not None:
            vis_path = args.output_dir / f"enhanced_{result['frame_file'].name}"
            create_enhanced_visualization(frame, template, result, vis_path)

    # Identify real matches
    analysis = identify_real_matches(results)

    print("\nReal Match Analysis:")
    print(f"Matches above adaptive threshold: {len(analysis['high_confidence_matches'])}")
    print(f"Grouped consecutive matches: {len(analysis['grouped_matches'])}")

    if analysis["grouped_matches"]:
        print("\nLikely real match sequences:")
        for i, group in enumerate(analysis["grouped_matches"]):
            print(f"  Group {i + 1}: {len(group)} frames")
            print(f"    Frames: {[r['frame_index'] + 1 for r in group]}")
            confidences_str = [f"{r['best_confidence']:.3f}" for r in group]
            print(f"    Confidences: {confidences_str}")
            print(f"    Average confidence: {np.mean([r['best_confidence'] for r in group]):.3f}")

            # Calculate timestamps (assuming 60 frame skip, 30 FPS)
            timestamps = [(r["frame_index"] * 60) / 30.0 for r in group]  # seconds
            timestamps_str = [f"{t:.1f}s" for t in timestamps]
            print(f"    Timestamps: {timestamps_str}")
            print()

    # Generate recommendations
    print("\nRecommendations for main.py:")
    if analysis["grouped_matches"]:
        best_group = max(
            analysis["grouped_matches"], key=lambda g: np.mean([r["best_confidence"] for r in g])
        )
        avg_confidence = np.mean([r["best_confidence"] for r in best_group])
        print(f"1. Use threshold around {avg_confidence - 0.05:.2f} to {avg_confidence - 0.1:.2f}")

        # Check scales used in best matches
        scales_used = [r["best_match"]["scale"] for r in best_group]
        common_scale = max(set(scales_used), key=scales_used.count)
        print(f"2. Target appears at {common_scale:.2f}x scale most commonly")

        # Check if we need multi-scale
        unique_scales = set(scales_used)
        if len(unique_scales) > 1:
            print(f"3. Consider multi-scale matching (scales used: {sorted(unique_scales)})")
        else:
            print(f"3. Single scale matching at {common_scale:.2f}x should work")
    else:
        print("1. Current threshold of 0.8 might be too high")
        print(f"2. Try threshold around {analysis['adaptive_threshold']:.2f}")
        print("3. Consider that the image might not actually appear clearly in the video")


if __name__ == "__main__":
    main()
