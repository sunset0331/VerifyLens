# data/

This directory holds the dataset used for VLM fine-tuning.

## synthetic/
Auto-generated using `scripts/generate_synthetic_data.py`.
Contains synthetic ID card images (Aadhaar/PAN/Passport) with programmatically
rendered fake names, DOBs, and document numbers — **no real PII**.

```
data/synthetic/
├── images/
│   ├── 00000_aadhaar.jpg
│   ├── 00001_pan.jpg
│   └── ...
└── dataset.jsonl      # {"image_path", "doc_type", "question", "answer"}
```

## Generating the dataset

```bash
pip install faker Pillow
python scripts/generate_synthetic_data.py --num_samples 5000 --output data/synthetic
```

## real_samples/
Not tracked in git. Contains a small set of real publicly-available
document sample images used for qualitative evaluation only.
