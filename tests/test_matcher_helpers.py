"""Unit tests for the pure helpers of the multi-scale matcher."""

import pytest

from victorious.matcher import extract_detection_times, frame_index_to_timestamp, parse_scales


class TestParseScales:
    def test_default_scale_string(self) -> None:
        assert parse_scales("0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2") == [
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
            1.1,
            1.2,
        ]

    def test_single_scale(self) -> None:
        assert parse_scales("1.0") == [1.0]

    def test_whitespace_is_stripped(self) -> None:
        assert parse_scales("0.5, 0.6 ,  0.7") == [0.5, 0.6, 0.7]

    def test_empty_entries_are_ignored(self) -> None:
        assert parse_scales("0.5,,0.6,") == [0.5, 0.6]

    def test_empty_string_yields_empty_list(self) -> None:
        assert parse_scales("") == []
        assert parse_scales("   ") == []

    def test_invalid_entry_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="could not convert string to float"):
            parse_scales("0.5,abc,0.7")


class TestFrameIndexToTimestamp:
    def test_zero_frame_is_zero_seconds(self) -> None:
        assert frame_index_to_timestamp(0, 2.0) == 0.0

    def test_basic_conversion(self) -> None:
        assert frame_index_to_timestamp(4, 2.0) == 2.0

    def test_fractional_fps(self) -> None:
        assert frame_index_to_timestamp(3, 1.5) == 2.0

    @pytest.mark.parametrize(
        ("frame_index", "fps", "expected"),
        [(0, 30.0, 0.0), (30, 30.0, 1.0), (15, 30.0, 0.5), (90, 30.0, 3.0)],
    )
    def test_parametrized(self, frame_index: int, fps: float, expected: float) -> None:
        assert frame_index_to_timestamp(frame_index, fps) == expected


class TestExtractDetectionTimes:
    def test_empty_input(self) -> None:
        assert extract_detection_times([]) == []

    def test_no_matches(self) -> None:
        assert extract_detection_times([(False, 0.0), (False, 0.5)]) == []

    def test_single_run_emits_only_start(self) -> None:
        assert extract_detection_times([(True, 0.0), (True, 0.5), (True, 1.0)]) == [0.0]

    def test_multiple_runs(self) -> None:
        seq = [
            (False, 0.0),
            (True, 0.5),
            (True, 1.0),
            (False, 1.5),
            (True, 2.0),
            (False, 2.5),
            (True, 3.0),
        ]
        assert extract_detection_times(seq) == [0.5, 2.0, 3.0]

    def test_match_at_first_frame(self) -> None:
        assert extract_detection_times([(True, 0.0), (False, 0.5), (True, 1.0)]) == [0.0, 1.0]

    def test_single_frame_gap_restarts_run(self) -> None:
        seq = [(True, 0.0), (False, 0.5), (True, 1.0)]
        assert extract_detection_times(seq) == [0.0, 1.0]
