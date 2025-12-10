"""Setup script for Unified Test Harness"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent / "unified_test_harness" / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

setup(
    name="unified-test-harness",
    version="1.0.0",
    description="Framework-agnostic, vector database-driven testing harness",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="SWORDIntel",
    packages=find_packages(),
    install_requires=[
        "pytest>=7.0.0",
        "pytest-cov>=4.0.0",
        "coverage>=7.0.0",
    ],
    extras_require={
        "vector": ["chromadb>=0.4.0"],
        "llm": ["openai>=1.0.0", "anthropic>=0.7.0"],
        "all": ["chromadb>=0.4.0", "openai>=1.0.0", "anthropic>=0.7.0"],
    },
    entry_points={
        "console_scripts": [
            "unified-test-harness=unified_test_harness.cli:main",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Testing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
