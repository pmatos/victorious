"""Output files are UTF-8 with LF line ends on every platform, whatever the system code page."""

from pathlib import Path

from victorious.cli import _emit


def test_a_transcript_in_portuguese_is_written_as_utf8_with_lf_line_ends(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"

    _emit('{"text": "Boa jogada, só restam 10"}\n', output)

    assert output.read_bytes() == '{"text": "Boa jogada, só restam 10"}\n'.encode()
