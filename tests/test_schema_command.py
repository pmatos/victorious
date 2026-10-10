"""Behaviour of `victorious schema`."""

import json
from pathlib import Path

import jsonschema

from tests.cli_runner import run_cli


def test_schema_prints_a_json_schema_with_its_version() -> None:
    result = run_cli("schema")

    assert result.returncode == 0, result.stderr
    schema = json.loads(result.stdout)
    jsonschema.Draft202012Validator.check_schema(schema)
    assert schema["schema_version"] == "1.0"
    assert schema["properties"]["schema_version"]["const"] == "1.0"
    assert {"schema_version", "recording", "matches"} <= set(schema["required"])


def test_schema_rejects_documents_missing_required_fields() -> None:
    schema = json.loads(run_cli("schema").stdout)
    broken = {"recording": {"path": "x.mkv", "duration": 1.0}, "matches": []}

    validator = jsonschema.Draft202012Validator(schema)

    assert not validator.is_valid(broken)


def test_schema_output_option_writes_the_file(tmp_path: Path) -> None:
    target = tmp_path / "schema.json"

    result = run_cli("schema", "-o", str(target))

    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert json.loads(target.read_text())["schema_version"] == "1.0"
