"""Property-based tests for the numeric core, using Hypothesis."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from main import extract_detection_times, frame_index_to_timestamp, parse_scales

finite_floats = st.floats(allow_nan=False, allow_infinity=False)


@given(st.text())
def test_parse_scales_never_raises_unexpectedly(text: str) -> None:
    """On arbitrary input, parse_scales returns a list of floats or raises ValueError."""
    try:
        result = parse_scales(text)
    except ValueError:
        return
    assert isinstance(result, list)
    assert all(isinstance(value, float) for value in result)


@given(st.lists(finite_floats.filter(lambda v: v == v), min_size=1))
def test_parse_scales_roundtrips_repr(values: list[float]) -> None:
    """Any list of finite floats survives a repr()->join->parse round-trip."""
    encoded = ",".join(repr(value) for value in values)
    assert parse_scales(encoded) == values


@given(
    st.integers(min_value=0, max_value=10**9),
    st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_timestamp_matches_division(frame_index: int, fps: float) -> None:
    timestamp = frame_index_to_timestamp(frame_index, fps)
    assert timestamp == pytest.approx(frame_index / fps)
    assert timestamp >= 0.0


@given(st.lists(st.tuples(st.booleans(), finite_floats)))
def test_extract_detection_times_properties(seq: list[tuple[bool, float]]) -> None:
    result = extract_detection_times(seq)
    match_timestamps = [timestamp for is_match, timestamp in seq if is_match]

    # Never emits more starts than there are matching frames.
    assert len(result) <= len(match_timestamps)

    # Every emitted timestamp is the timestamp of a matching frame.
    assert all(timestamp in match_timestamps for timestamp in result)

    # Exactly one start per contiguous run of matches (False->True transitions).
    transitions = 0
    previously_matching = False
    for is_match, _ in seq:
        if is_match and not previously_matching:
            transitions += 1
        previously_matching = is_match
    assert len(result) == transitions

    # The first start is never earlier in the sequence than the first match.
    if result:
        assert result[0] == match_timestamps[0]
