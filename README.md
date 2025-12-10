# Unified Test Harness

A framework-agnostic, vector database-driven testing harness that ci devised for rapid test generation in huge programs. Works with Python, C, and Rust projects and testing frameworks.

## Overview

This unified test harness provides:

- **Multi-Language Support**: Works with Python, C, and Rust projects
- **Framework Agnostic**: Works with pytest, unittest, Unity (C), cargo test (Rust), and custom test frameworks
- **Vector Database Integration**: Uses ChromaDB for semantic code similarity search
- **LLM-Powered Generation**: Supports OpenAI and Anthropic for intelligent test generation
- **Coverage-Driven**: Identifies gaps and prioritizes test generation
- **Template-Based**: Learns from existing tests and adapts them to new code
- **Configurable**: Supports different project structures (standard, src/, modules/, C projects, Rust projects)
- **CI/CD Ready**: Designed for integration into automated workflows

## Quick Start

### Python Projects

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

### C Projects

```bash
# Install system dependencies
sudo apt-get install gcc lcov  # or equivalent for your system

# Initialize harness (auto-detects C project)
python -m unified_test_harness.cli --init --language c

# Run full harness
python -m unified_test_harness.cli --language c
```

### Rust Projects

```bash
# Install Rust and cargo-tarpaulin
cargo install cargo-tarpaulin

# Initialize harness (auto-detects Rust project)
python -m unified_test_harness.cli --init --language rust

# Run full harness
python -m unified_test_harness.cli --language rust
```

## Features

### 1. Coverage Analysis
- **Python**: Uses pytest-cov for coverage analysis
- **C**: Uses gcov/lcov for coverage analysis
- **Rust**: Uses cargo-tarpaulin for coverage analysis
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
- Creates test code in multiple formats:
  - **Python**: pytest and unittest formats
  - **C**: Unity test framework format
  - **Rust**: Built-in `#[test]` format
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

The harness supports multiple project structure types:

### Python Projects
1. **standard**: Flat structure with tests/ directory
2. **src_layout**: Source code in src/, tests in tests/
3. **modules_layout**: Source code in modules/, tests in tests/

### C Projects
- **c_project**: Standard C project with src/ and tests/ directories
- Supports Unity, CUnit, and Check test frameworks
- Uses gcov/lcov for coverage

### Rust Projects
- **rust_project**: Standard Cargo project structure
- Uses built-in `#[test]` framework
- Uses cargo-tarpaulin for coverage

Language and project type are auto-detected, but can be explicitly specified:
```bash
python -m unified_test_harness.cli --language rust --project-type rust_project
```

## Output

After running, the harness generates:

- `tests/harness_output/HARNESS_SUMMARY.md` - Summary report
- `tests/harness_output/test_vectors.json` - Generated test vectors
- `tests/harness_output/generated_tests/` - Executable test code
- `tests/harness_output/coverage.xml` - Coverage report
- `tests/harness_output/htmlcov/` - HTML coverage report

## Requirements

### Core Requirements
- Python 3.8+
- pytest (for Python test execution)
- pytest-cov (for Python coverage)
- chromadb (for vector database, optional)
- openai or anthropic (for LLM generation, optional)

### C/C++ Requirements (optional)
- gcc compiler
- gcov (usually included with gcc)
- lcov (for coverage reports)
- Unity test framework (recommended) or CUnit/Check

### Rust Requirements (optional)
- Rust and Cargo
- cargo-tarpaulin (for coverage): `cargo install cargo-tarpaulin`

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
