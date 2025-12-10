"""
Unified Test Harness

A framework-agnostic, vector database-driven testing harness that combines
coverage analysis, LLM-powered test generation, and template-based test creation.

Works with any Python project and testing framework.
"""

__version__ = "1.0.0"

from .test_vector import TestVector, TestVectorType, TestPriority, TestVectorRegistry
from .coverage_analyzer import CoverageAnalyzer
from .code_embedder import CodeEmbedder
from .llm_generator import LLMTestGenerator
from .harness_runner import TestHarnessRunner
from .config import HarnessConfig

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
]
