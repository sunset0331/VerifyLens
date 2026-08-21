"""
tests/test_ocr_llm_extractor.py
-------------------------------
Unit tests for the OCRLLMExtractor and adapter loading logic.
"""

import pytest
from unittest.mock import patch, MagicMock
from src.document.ocr_llm_extractor import OCRLLMExtractor

class TestOCRLLMExtractor:

    @patch("mlx_lm.load")
    @patch("pathlib.Path.exists", return_value=True)
    def test_loads_adapter_if_exists(self, mock_exists, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        extractor = OCRLLMExtractor(
            model_path="test-model",
            adapter_path="fake-adapter-path"
        )
        assert extractor._using_adapter is True
        mock_load.assert_called_with("test-model", adapter_path="fake-adapter-path")

    @patch("mlx_lm.load")
    @patch("pathlib.Path.exists", return_value=False)
    def test_raises_error_if_adapter_missing(self, mock_exists, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        with pytest.raises(FileNotFoundError, match="Adapter 'fake-adapter-path' not found"):
            OCRLLMExtractor(
                model_path="test-model",
                adapter_path="fake-adapter-path"
            )

    @patch("mlx_lm.load")
    @patch("pathlib.Path.exists", return_value=False)
    def test_loads_base_model_if_adapter_none(self, mock_exists, mock_load):
        mock_load.return_value = (MagicMock(), MagicMock())
        extractor = OCRLLMExtractor(
            model_path="test-model",
            adapter_path=None
        )
        assert extractor._using_adapter is False
        mock_load.assert_called_with("test-model")
