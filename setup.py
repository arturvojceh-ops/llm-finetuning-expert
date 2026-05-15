"""
Setup script for LLM Fine-Tuning Expert Suite

Author: LLM Fine-Tuning Expert
License: MIT
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = f.read().strip().split('\n')

# Read version
version = "1.0.0"

setup(
    name="llm-finetuning-expert",
    version=version,
    author="LLM Fine-Tuning Expert",
    author_email="expert@llm-finetuning.com",
    description="Production-Grade LLM Fine-Tuning Framework with 2026 SOTA Techniques",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/llm-finetuning-expert",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=98.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-finetuning-trainer=training.advanced_trainer:main",
            "llm-inference-engine=inference.optimized_inference:main",
            "benchmark=inference.optimized_inference:benchmark_model",
        ],
    },
    include_package_data=True,
    package_data={
        "llm_finetuning_expert": [
            "src/training/*",
            "src/inference/", 
            "src/data/*",
            "src/monitoring/*",
            "src/optimization/*",
            "examples/*",
            "docs/*",
        ],
    },
    keywords=[
        "llm", "fine-tuning", "maching learning", "artificial intelligence",
        "transformers", "pytorch", "peft", "lora", "qlora",
        "adalora", "dora", "oft", "pissa", "muon", "galore",
        "badam", "apollo", "adam-mini", "optimization",
        "inference", "quantization", "4bit", "8bit", "fp8",
        "vllm", "sglang","unsloth", "deeppeek", "reasoning",
        "enterprise", "production", "mlops", "monitoring", "deployment"
    ],
    project_urls={
        "Bug Reports": "https://github.com/yourusername/llm-finetuning-expert/issues",
        "Source": "https://github.com/yourusername/llm-finetuning-expert",
        "Documentation": "https://llm-finetuning-expert.readthedocs.io/",
    },
)
