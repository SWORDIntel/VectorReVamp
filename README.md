# Unified Test Harness

A framework-agnostic, vector database-driven testing harness that ci devised for rapid test generation in huge programs. Works with any Python project and testing framework.

## Overview

This unified test harness provides:

- **Framework Agnostic**: Works with pytest, unittest, nose, and custom test frameworks
- **Vector Database Integration**: Uses ChromaDB for semantic code similarity search
- **LLM-Powered Generation**: Supports OpenAI and Anthropic for intelligent test generation
- **Coverage-Driven**: Identifies gaps and prioritizes test generation
- **Template-Based**: Learns from existing tests and adapts them to new code
- **Configurable**: Supports different project structures (standard, src/, modules/)
- **CI/CD Ready**: Designed for integration into automated workflows

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize harness
python -m unified_test_harness.cli --init

# Run full harness
python -m unified_test_harness.cli

# With LLM generation
export OPENAI_API_KEY=your_key_here
python -m unified_test_harness.cli --use-llm
```

## Features

### 1. Coverage Analysis
- Runs test suite with coverage tracking
- Identifies uncovered functions and modules
- Prioritizes gaps by severity and importance
- Generates comprehensive coverage reports

### 2. Vector Database Integration
- Embeds source code into vector space
- Learns from existing test patterns
- Finds similar code for template adaptation
- Enables semantic similarity search

### 3. LLM-Powered Generation
- Generates comprehensive test vectors
- Creates test code in multiple formats
- Supports OpenAI and Anthropic APIs
- Falls back to template-based generation

### 4. Template-Based Generation
- Extracts patterns from existing tests
- Adapts templates to new code
- Generates parametrized tests
- Maintains consistency with existing tests

## Project Structure

```
unified_test_harness/
├── __init__.py              # Package initialization
├── config.py                # Configuration system
├── test_vector.py           # Test vector definitions
├── coverage_analyzer.py     # Coverage analysis
├── code_embedder.py         # Vector database embedding
├── llm_generator.py         # LLM-powered generation
├── harness_runner.py        # Main orchestrator
├── conftest.py              # Pytest integration
├── cli.py                   # Command-line interface
└── README.md                 # Detailed documentation
```

## Usage

### Command Line

```bash
# Initialize vector database
python -m unified_test_harness.cli --init

# Run coverage analysis
python -m unified_test_harness.cli --coverage-only

# Generate tests
python -m unified_test_harness.cli --generate-only

# Full harness with LLM
python -m unified_test_harness.cli --use-llm --llm-provider openai

# Focus on specific modules
python -m unified_test_harness.cli --modules router orchestrator
```

### Python API

```python
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

# Create configuration
config = HarnessConfig.create_for_project(Path("."), "standard")
config.llm_enabled = True
config.llm_provider = "openai"

# Create runner
runner = TestHarnessRunner(config)

# Run full harness
results = runner.run_full_harness()

print(f"Coverage: {results['coverage_report']['coverage_percentage']:.2f}%")
print(f"Generated: {len(results['generated_vectors'])} vectors")
```

## Configuration

The harness supports three project structure types:

1. **standard**: Flat structure with tests/ directory
2. **src_layout**: Source code in src/, tests in tests/
3. **modules_layout**: Source code in modules/, tests in tests/

## Output

After running, the harness generates:

- `tests/harness_output/HARNESS_SUMMARY.md` - Summary report
- `tests/harness_output/test_vectors.json` - Generated test vectors
- `tests/harness_output/generated_tests/` - Executable test code
- `tests/harness_output/coverage.xml` - Coverage report
- `tests/harness_output/htmlcov/` - HTML coverage report

## Requirements

- Python 3.8+
- pytest (for test execution)
- pytest-cov (for coverage)
- chromadb (for vector database, optional)
- openai or anthropic (for LLM generation, optional)

## Installation

```bash
# Install from source
pip install -e .

# Or install with extras
pip install -e ".[all]"  # Includes vector DB and LLM support
```

## Documentation

- **[README.md](unified_test_harness/README.md)** - Detailed documentation
- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[setup.py](setup.py)** - Installation script

## Based On

This unified harness combines methodologies from:

- **KP14 Vector Database Testing Harness**: ChromaDB integration, template-based generation
- **GETMOVIN Test Harness**: LLM integration, pytest compatibility, test vector registry

## License

Same as main project (for authorized security testing only).

## Contributing

When adding support for new frameworks:

1. Extend `FrameworkConfig` in `config.py`
2. Add test code generator in `llm_generator.py`
3. Update coverage command patterns
4. Add framework-specific fixtures in `conftest.py`

## Support

For issues or questions:
- Check `unified_test_harness/README.md` for detailed documentation
- Review `QUICKSTART.md` for quick start guide
- Check generated test code for adaptation patterns
