from setuptools import setup, find_packages

setup(
    name="verifylens",
    version="0.1.0",
    description="Multimodal KYC verification pipeline: document intelligence + VLM extraction + face verification",
    author="Utkarsh Gaur",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "peft>=0.12.0",
        "fastapi>=0.115.0",
        "Pillow>=10.0.0",
        "numpy>=1.24.0",
    ],
)
