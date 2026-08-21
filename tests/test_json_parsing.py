"""
tests/test_json_parsing.py
---------------------------
Unit tests for the robust JSON parsing used by both extractors.
No model weights required.
"""

import pytest
from src.document.ocr_llm_extractor import _parse_json, _validate_fields
from src.document.schema import DocumentFields


class TestParseJson:

    def test_plain_json(self):
        raw = '{"name": "Ravi Kumar", "dob": "14/09/1990"}'
        parsed, valid, err = _parse_json(raw)
        assert valid is True
        assert err is None
        assert parsed["name"] == "Ravi Kumar"

    def test_markdown_fenced_json(self):
        raw = '```json\n{"name": "Priya", "dob": "01/01/1985"}\n```'
        parsed, valid, err = _parse_json(raw)
        assert valid is True
        assert parsed["name"] == "Priya"

    def test_markdown_fenced_no_lang(self):
        raw = '```\n{"doc_type": "PAN Card"}\n```'
        parsed, valid, err = _parse_json(raw)
        assert valid is True
        assert parsed["doc_type"] == "PAN Card"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the extracted info: {"name": "Arjun"} Hope that helps!'
        parsed, valid, err = _parse_json(raw)
        assert valid is True
        assert parsed["name"] == "Arjun"

    def test_json_with_null_values(self):
        raw = '{"name": "Ravi", "dob": null, "doc_number": null}'
        parsed, valid, err = _parse_json(raw)
        assert valid is True
        assert parsed["name"] == "Ravi"
        assert parsed["dob"] is None

    def test_empty_json_object(self):
        raw = "{}"
        parsed, valid, err = _parse_json(raw)
        assert valid is True
        assert parsed == {}

    def test_completely_invalid_no_braces(self):
        raw = "I cannot extract any information from this image."
        parsed, valid, err = _parse_json(raw)
        assert valid is False
        assert parsed is None
        assert err is not None

    def test_malformed_json_in_braces(self):
        raw = "{name: Ravi, dob: 1990}"  # missing quotes
        parsed, valid, err = _parse_json(raw)
        assert valid is False

    def test_empty_string(self):
        parsed, valid, err = _parse_json("")
        assert valid is False
        assert parsed is None

    def test_whitespace_only(self):
        parsed, valid, err = _parse_json("   \n  ")
        assert valid is False


class TestValidateFields:

    def test_valid_dict_maps_correctly(self):
        d = {"name": "Ravi Kumar", "dob": "14/09/1990", "doc_type": "Aadhaar Card"}
        fields = _validate_fields(d)
        assert isinstance(fields, DocumentFields)
        assert fields.name == "Ravi Kumar"
        assert fields.dob == "14/09/1990"

    def test_none_input_returns_empty(self):
        fields = _validate_fields(None)
        assert fields.is_empty()

    def test_extra_keys_ignored(self):
        d = {"name": "Ravi", "unknown_key": "should_be_ignored"}
        fields = _validate_fields(d)
        assert fields.name == "Ravi"
        assert not hasattr(fields, "unknown_key")

    def test_missing_keys_become_none(self):
        d = {"name": "Ravi"}
        fields = _validate_fields(d)
        assert fields.dob is None
        assert fields.doc_number is None
        assert fields.address is None

    def test_null_value_in_dict(self):
        d = {"name": "Ravi", "dob": None}
        fields = _validate_fields(d)
        assert fields.dob is None

    def test_empty_string_normalized_to_none(self):
        d = {"name": "", "dob": "  "}
        fields = _validate_fields(d)
        assert fields.name is None
        assert fields.dob is None

    def test_all_fields_populated(self):
        d = {
            "name": "Priya Sharma",
            "dob": "01/01/1985",
            "doc_number": "ABCDE1234F",
            "doc_type": "PAN Card",
            "gender": "Female",
            "address": "New Delhi",
        }
        fields = _validate_fields(d)
        assert not fields.is_empty()
        assert len(fields.non_null_fields()) == 6
