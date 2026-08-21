"""
tests/test_vlm_extractor.py
-----------------------------
Unit + integration tests for VLMExtractor (Approach 2).

Unit tests use mocks — no model weights required.
The integration/smoke test actually loads Qwen2.5-VL and runs
a real inference pass. It is marked @pytest.mark.integration
and skipped by default unless explicitly requested.

Run unit tests only:
    .venv312/bin/python -m pytest tests/test_vlm_extractor.py -m "not integration" -v

Run with integration (downloads ~1.5 GB on first run):
    .venv312/bin/python -m pytest tests/test_vlm_extractor.py -m integration -v -s
"""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from src.document.schema import DocumentFields, ExtractionResult


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def blank_image():
    """1×1 white PIL image — valid input, will produce no useful extraction."""
    return Image.new("RGB", (1, 1), color=(255, 255, 255))


@pytest.fixture
def mock_vlm_extractor():
    """VLMExtractor with mocked mlx_vlm.load — no weights downloaded."""
    mock_model = MagicMock()
    mock_model.config = MagicMock()
    mock_processor = MagicMock()

    with patch("mlx_vlm.load", return_value=(mock_model, mock_processor)):
        from src.document.vlm.vlm_extractor import VLMExtractor
        extractor = VLMExtractor(model_path="mock-model")
    return extractor, mock_model, mock_processor


# ── Unit tests (no weights) ───────────────────────────────────────────────────

class TestVLMExtractorInit:

    def test_extraction_mode_is_vlm(self, mock_vlm_extractor):
        extractor, _, _ = mock_vlm_extractor
        assert extractor.extraction_mode == "vlm"

    def test_model_name_matches_path(self, mock_vlm_extractor):
        extractor, _, _ = mock_vlm_extractor
        assert extractor.model_name == "mock-model"


class TestVLMExtractorImagePreprocessing:

    def test_rgb_conversion(self, mock_vlm_extractor):
        extractor, _, _ = mock_vlm_extractor
        rgba_image = Image.new("RGBA", (100, 100))
        result = extractor._preprocess_image(rgba_image)
        assert result.mode == "RGB"

    def test_large_image_resized(self, mock_vlm_extractor):
        extractor, _, _ = mock_vlm_extractor
        large_image = Image.new("RGB", (4000, 3000))
        result = extractor._preprocess_image(large_image)
        assert max(result.size) <= extractor._resize_max_px

    def test_small_image_not_resized(self, mock_vlm_extractor):
        extractor, _, _ = mock_vlm_extractor
        small_image = Image.new("RGB", (400, 300))
        result = extractor._preprocess_image(small_image)
        assert result.size == (400, 300)

    def test_aspect_ratio_preserved_on_resize(self, mock_vlm_extractor):
        extractor, _, _ = mock_vlm_extractor
        image = Image.new("RGB", (4480, 2240))  # 2:1 aspect ratio
        result = extractor._preprocess_image(image)
        w, h = result.size
        ratio = w / h
        assert abs(ratio - 2.0) < 0.01


class TestVLMExtractorExtract:

    def _make_extract_result(self, mock_vlm_extractor, generate_return: str, blank_image):
        extractor, mock_model, mock_processor = mock_vlm_extractor

        with (
            patch("mlx_vlm.generate", return_value=generate_return),
            patch("mlx_vlm.prompt_utils.apply_chat_template", return_value="mock_prompt"),
        ):
            return extractor.extract(blank_image)

    def test_valid_json_response(self, mock_vlm_extractor, blank_image):
        raw = '{"name": "Ravi Kumar", "dob": "14/09/1990", "doc_number": "1234 5678 9012", "doc_type": "Aadhaar Card", "gender": "Male", "address": null}'
        result = self._make_extract_result(mock_vlm_extractor, raw, blank_image)

        assert isinstance(result, ExtractionResult)
        assert result.json_valid is True
        assert result.fields.name == "Ravi Kumar"
        assert result.fields.address is None
        assert result.extraction_mode == "vlm"
        assert result.extractor_fallback is False

    def test_fenced_json_response(self, mock_vlm_extractor, blank_image):
        raw = '```json\n{"name": "Priya", "doc_type": "PAN Card"}\n```'
        result = self._make_extract_result(mock_vlm_extractor, raw, blank_image)
        assert result.json_valid is True
        assert result.fields.name == "Priya"

    def test_invalid_json_response(self, mock_vlm_extractor, blank_image):
        raw = "I cannot extract information from this image."
        result = self._make_extract_result(mock_vlm_extractor, raw, blank_image)
        assert result.json_valid is False
        assert result.parse_error is not None
        assert result.fields.is_empty()

    def test_extra_fields_ignored(self, mock_vlm_extractor, blank_image):
        raw = '{"name": "Test", "invented_field": "should_be_dropped"}'
        result = self._make_extract_result(mock_vlm_extractor, raw, blank_image)
        assert result.json_valid is True
        assert result.fields.name == "Test"
        assert not hasattr(result.fields, "invented_field")

    def test_latency_ms_present(self, mock_vlm_extractor, blank_image):
        raw = '{"name": "Test"}'
        result = self._make_extract_result(mock_vlm_extractor, raw, blank_image)
        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    def test_generation_exception_returns_error_result(self, mock_vlm_extractor, blank_image):
        extractor, _, _ = mock_vlm_extractor
        with (
            patch("mlx_vlm.generate", side_effect=RuntimeError("GPU OOM")),
            patch("mlx_vlm.prompt_utils.apply_chat_template", return_value="prompt"),
        ):
            result = extractor.extract(blank_image)
        assert result.error is not None
        assert "GPU OOM" in result.error
        assert result.fields.is_empty()


# ── Integration/smoke test (actual model weights) ─────────────────────────────

@pytest.mark.integration
def test_vlm_extractor_real_model_smoke():
    """
    Smoke test that loads Qwen2.5-VL-2B-Instruct-4bit and runs one
    inference pass on a synthetic 800×500 white image.

    Downloads ~1.5 GB on first run (cached subsequently).
    """
    import time
    from src.document.vlm.vlm_extractor import VLMExtractor

    print("\n[Integration] Loading VLMExtractor with real model weights...")
    t_load_start = time.perf_counter()
    extractor = VLMExtractor()
    t_load_end = time.perf_counter()
    print(f"[Integration] Model load time: {(t_load_end - t_load_start):.1f}s")

    # Create a blank white image (no document content — will likely return all-null)
    image = Image.new("RGB", (800, 500), color=(255, 255, 255))

    # Warm-up pass (first inference is slower due to compilation)
    print("[Integration] Warm-up inference...")
    _ = extractor.extract(image)

    # Timed inference
    print("[Integration] Timed inference...")
    t0 = time.perf_counter()
    result = extractor.extract(image)
    t1 = time.perf_counter()
    wall_ms = (t1 - t0) * 1000

    print(f"[Integration] Wall latency: {wall_ms:.0f} ms")
    print(f"[Integration] Model latency: {result.latency_ms} ms")
    print(f"[Integration] JSON valid: {result.json_valid}")
    print(f"[Integration] Raw output: {result.raw_output!r}")
    print(f"[Integration] Parsed fields: {result.fields.to_dict()}")

    # Structural assertions (do not require meaningful output on blank image)
    assert isinstance(result, ExtractionResult)
    assert result.extraction_mode == "vlm"
    assert result.extractor_fallback is False
    assert result.model == "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    assert isinstance(result.fields, DocumentFields)
    # Latency should be measured
    assert result.latency_ms is not None
    assert result.latency_ms >= 0
    # We do NOT assert json_valid=True on a blank image
    # (the model may legitimately refuse or return malformed output)
