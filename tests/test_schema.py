"""
tests/test_schema.py
----------------------
Unit tests for the common DocumentFields schema.
No model weights required.
"""

import pytest
from src.document.schema import DocumentFields, ExtractionResult


class TestDocumentFields:

    def test_all_fields_none_by_default(self):
        f = DocumentFields()
        assert f.name is None
        assert f.dob is None
        assert f.doc_number is None
        assert f.doc_type is None
        assert f.gender is None
        assert f.address is None

    def test_is_empty_when_all_none(self):
        assert DocumentFields().is_empty()

    def test_not_empty_when_one_field_set(self):
        f = DocumentFields(name="Ravi Kumar")
        assert not f.is_empty()

    def test_empty_string_becomes_none(self):
        f = DocumentFields(name="", dob="  ")
        assert f.name is None
        assert f.dob is None

    def test_non_null_fields_filters_none(self):
        f = DocumentFields(name="Ravi Kumar", dob=None)
        nonnull = f.non_null_fields()
        assert "name" in nonnull
        assert "dob" not in nonnull

    def test_to_dict_includes_all_keys(self):
        f = DocumentFields(name="Test")
        d = f.to_dict()
        assert set(d.keys()) == {"name", "dob", "doc_number", "doc_type", "gender", "address"}

    def test_unknown_keys_ignored(self):
        """Pydantic should ignore extra keys silently."""
        f = DocumentFields(**{"name": "Test", "unknown_field": "ignored"})
        assert not hasattr(f, "unknown_field")

    def test_full_document(self):
        f = DocumentFields(
            name="Ravi Kumar",
            dob="14/09/1990",
            doc_number="1234 5678 9012",
            doc_type="Aadhaar Card",
            gender="Male",
            address="123 Main Street, Mumbai",
        )
        assert f.name == "Ravi Kumar"
        assert not f.is_empty()
        assert len(f.non_null_fields()) == 6


class TestExtractionResult:

    def _make_result(self, **kwargs):
        defaults = {
            "extraction_mode": "vlm",
            "model": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
            "json_valid": True,
        }
        defaults.update(kwargs)
        return ExtractionResult(**defaults)

    def test_default_fields_empty(self):
        r = self._make_result()
        assert r.fields.is_empty()

    def test_json_valid_false_preserved(self):
        r = self._make_result(json_valid=False, parse_error="No JSON found")
        assert not r.json_valid
        assert r.parse_error == "No JSON found"

    def test_extraction_mode_stored(self):
        r = self._make_result(extraction_mode="ocr_llm")
        assert r.extraction_mode == "ocr_llm"

    def test_extractor_fallback_default_false(self):
        r = self._make_result()
        assert r.extractor_fallback is False

    def test_error_field(self):
        r = self._make_result(json_valid=False, error="VLM crashed")
        assert r.error == "VLM crashed"

    def test_latency_ms(self):
        r = self._make_result(latency_ms=1234.5)
        assert r.latency_ms == 1234.5
