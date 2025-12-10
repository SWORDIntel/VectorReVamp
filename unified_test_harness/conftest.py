"""
Pytest configuration for unified test harness

Provides fixtures for pytest integration.
"""

import pytest
from pathlib import Path
from .test_vector import TestVectorRegistry
from .config import HarnessConfig


@pytest.fixture(scope="session")
def harness_config(source_root):
    """Load harness configuration"""
    return HarnessConfig.create_for_project(source_root)


@pytest.fixture(scope="session")
def vector_registry(harness_config):
    """Load test vector registry"""
    registry = TestVectorRegistry()
    registry_file = harness_config.output_dir / "test_vectors.json"
    if registry_file.exists():
        registry.load(registry_file)
    return registry


@pytest.fixture(scope="session")
def test_output_dir(harness_config):
    """Test output directory"""
    harness_config.output_dir.mkdir(parents=True, exist_ok=True)
    return harness_config.output_dir


@pytest.fixture(scope="session")
def source_root():
    """Source code root directory"""
    return Path(__file__).parent.parent.parent
