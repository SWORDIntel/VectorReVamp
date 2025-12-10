"""
Unified Test Harness

A framework-agnostic, vector database-driven testing harness that combines
coverage analysis, LLM-powered test generation, and template-based test creation.

Works with Python, C, and Rust projects and testing frameworks.
"""

__version__ = "1.0.0"

from .test_vector import TestVector, TestVectorType, TestPriority, TestVectorRegistry
from .coverage_analyzer import CoverageAnalyzer
from .code_embedder import CodeEmbedder
from .llm_generator import LLMTestGenerator
from .harness_runner import TestHarnessRunner
from .config import HarnessConfig
from .language_parser import LanguageParser, Language, CodeElement

__all__ = [
    'TestVector',
    'TestVectorType',
    'TestPriority',
    'TestVectorRegistry',
    'CoverageAnalyzer',
    'CodeEmbedder',
    'LLMTestGenerator',
    'TestHarnessRunner',
    'HarnessConfig',
    'LanguageParser',
    'Language',
    'CodeElement',
]
