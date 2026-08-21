"""
tests/test_extractor_router.py
-------------------------------
Unit tests for the extractor router.
Uses mocks so no model weights are required.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestExtractorRouter:

    def test_invalid_mode_raises_value_error(self):
        from src.document.extractor_router import get_extractor
        with pytest.raises(ValueError, match="Unknown extraction mode"):
            get_extractor(mode="invalid_mode")

    def test_supported_modes_constant(self):
        from src.document.extractor_router import SUPPORTED_MODES
        assert "ocr_llm" in SUPPORTED_MODES
        assert "vlm" in SUPPORTED_MODES

    @patch("pathlib.Path.exists", return_value=True)
    def test_ocr_llm_mode_returns_ocr_llm_extractor(self, mock_exists):
        from src.document.extractor_router import get_extractor
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template = MagicMock(return_value="prompt")

        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            extractor = get_extractor(mode="ocr_llm")

        from src.document.ocr_llm_extractor import OCRLLMExtractor
        assert isinstance(extractor, OCRLLMExtractor)
        assert extractor.extraction_mode == "ocr_llm"

    def test_vlm_mode_returns_vlm_extractor(self):
        from src.document.extractor_router import get_extractor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("mlx_vlm.load", return_value=(mock_model, mock_processor)):
            extractor = get_extractor(mode="vlm")

        from src.document.vlm.vlm_extractor import VLMExtractor
        assert isinstance(extractor, VLMExtractor)
        assert extractor.extraction_mode == "vlm"

    def test_mode_is_case_insensitive(self):
        """Router should accept 'VLM' the same as 'vlm'."""
        from src.document.extractor_router import get_extractor
        mock_model = MagicMock()
        mock_processor = MagicMock()

        with patch("mlx_vlm.load", return_value=(mock_model, mock_processor)):
            extractor = get_extractor(mode="VLM")
        from src.document.vlm.vlm_extractor import VLMExtractor
        assert isinstance(extractor, VLMExtractor)

    @patch("pathlib.Path.exists", return_value=True)
    def test_mode_strips_whitespace(self, mock_exists):
        from src.document.extractor_router import get_extractor
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)):
            extractor = get_extractor(mode="  ocr_llm  ")
        from src.document.ocr_llm_extractor import OCRLLMExtractor
        assert isinstance(extractor, OCRLLMExtractor)

