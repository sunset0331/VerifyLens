"""
verifylens_kaggle — Self-contained VerifyLens benchmark library for Kaggle GPU.

This package is intentionally standalone: it does NOT import from the
production src/ directory. Shared evaluation logic (normalizer, metrics)
is copied here and must remain synchronized with:

  src/evaluation/normalizer.py
  src/evaluation/metrics.py
"""
