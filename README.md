# Unified Test Harness

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Automatically generate comprehensive test suites for Python, C, and Rust projects using AI-powered analysis and coverage-driven test generation.**

The Unified Test Harness is an intelligent testing framework that analyzes your codebase, identifies coverage gaps, and generates high-quality test code. It combines vector database technology for pattern matching with LLM-powered generation to create tests that require minimal manual correction.

## ✨ Key Features

- 🤖 **AI-Powered Test Generation**: Uses OpenAI or Anthropic to generate intelligent, context-aware tests
- 📊 **Coverage-Driven**: Automatically identifies untested code and prioritizes test generation
- 🔍 **Pattern Learning**: Learns from existing tests and adapts patterns to new code
- 🌍 **Multi-Language Support**: Works with Python, C, and Rust projects
- 🎯 **Framework Agnostic**: Supports pytest, unittest, Unity (C), cargo test (Rust)
- 🚀 **CI/CD Ready**: Designed for integration into automated workflows
- 📈 **Smart Analysis**: Extracts function signatures, types, and dependencies for accurate test generation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd unified-test-harness

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Or install with all optional dependencies
pip install -e ".[all]"  # Includes vector DB and LLM support
```

### Basic Usage (Python Project)

```bash
# 1. Navigate to your project directory
cd /path/to/your/project

# 2. Initialize the harness (creates vector database)
python -m unified_test_harness.cli --init

# 3. Run the full harness (coverage analysis + test generation)
python -m unified_test_harness.cli

# 4. With AI-powered generation (requires API key)
export OPENAI_API_KEY=your_api_key_here
python -m unified_test_harness.cli --use-llm
```

### What Happens Next?

1. **Coverage Analysis**: The harness runs your existing tests and analyzes coverage
2. **Gap Identification**: Identifies functions and modules without tests
3. **Test Generation**: Creates test vectors and generates executable test code
4. **Output**: Saves generated tests to `tests/harness_output/generated_tests/`

## 📖 Detailed Usage Guide

### For Python Projects

```bash
# Standard Python project (auto-detected)
python -m unified_test_harness.cli --init
python -m unified_test_harness.cli

# With src/ layout
python -m unified_test_harness.cli --project-type src_layout --init
python -m unified_test_harness.cli --project-type src_layout

# Focus on specific modules
python -m unified_test_harness.cli --modules router database auth
```

### For C Projects

```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install gcc lcov

# Initialize and run (auto-detects C project)
python -m unified_test_harness.cli --init --language c
python -m unified_test_harness.cli --language c

# Or explicitly specify
python -m unified_test_harness.cli --language c --project-type c_project
```

**Note**: C projects should use Unity test framework. Ensure your Makefile includes coverage flags (`--coverage`).

### For Rust Projects

```bash
# Install cargo-tarpaulin for coverage
cargo install cargo-tarpaulin

# Initialize and run (auto-detects Rust project)
python -m unified_test_harness.cli --init --language rust
python -m unified_test_harness.cli --language rust
```

### For DSMILSystem (pytest, entrypoint focus)

```bash
# From DSMILSystem root
python -m unified_test_harness.cli --project-type dsmil_system --init
python -m unified_test_harness.cli --project-type dsmil_system --use-llm \
  --modules entrypoints.rce entrypoints.scan adapters.hdais
```

- Safety defaults: network/filesystem writes are blocked in generated tests unless explicitly enabled
- Targets entrypoints/adapters; outputs pytest suites tuned for minimal post-editing

## 🎯 Common Use Cases

### 1. Daily Development Workflow

```bash
# After writing new code, generate tests
python -m unified_test_harness.cli --coverage-only  # Check coverage
python -m unified_test_harness.cli --modules new_feature  # Generate tests
```

### 2. CI/CD Integration

```yaml
# .github/workflows/test-generation.yml
name: Generate Tests

on: [push, pull_request]

jobs:
  generate-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - run: pip install -r requirements.txt
      - run: pip install -e .
      - run: |
          export OPENAI_API_KEY=${{ secrets.OPENAI_API_KEY }}
          python -m unified_test_harness.cli --init
          python -m unified_test_harness.cli --use-llm
      - run: pytest tests/harness_output/generated_tests/
```

### 3. Python API Usage

```python
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

# Configure the harness
config = HarnessConfig.create_for_project(
    source_root=Path("."),
    project_type="standard"  # or "src_layout", "modules_layout"
)

# Enable AI-powered generation
config.llm_enabled = True
config.llm_provider = "openai"  # or "anthropic"
config.llm_api_key = "your-api-key-here"
config.llm_model = "gpt-4"  # or "gpt-3.5-turbo", "claude-3-opus"

# Create and run harness
runner = TestHarnessRunner(config)

# Run full workflow
results = runner.run_full_harness(
    use_llm=True,
    focus_modules=["router", "database"]  # Optional: focus on specific modules
)

# Access results
print(f"Coverage: {results['coverage_report']['coverage_percentage']:.2f}%")
print(f"Generated {len(results['generated_vectors'])} test vectors")
print(f"Tests saved to: {config.output_dir / 'generated_tests'}")
```

### 4. Advanced Configuration

```python
from unified_test_harness import HarnessConfig, TestHarnessRunner

config = HarnessConfig.create_for_project(Path("."), "standard")

# Coverage settings
config.coverage_threshold = 0.8  # Minimum coverage percentage
config.coverage_minimum = 0.0    # Minimum acceptable coverage

# Generation settings
config.batch_size = 50           # Process modules in batches
config.max_tests_per_module = 100 # Max tests per module

# Vector database (optional but recommended)
config.use_vector_db = True
config.vector_db_path = Path("./.harness_db")

# Test generation preferences
config.generate_unit_tests = True
config.generate_integration_tests = True
config.generate_edge_cases = True
config.generate_error_tests = True

# Output settings
config.save_generated_tests = True
config.test_output_format = "pytest"  # or "unittest"

runner = TestHarnessRunner(config)
results = runner.run_full_harness()
```

## 📋 Command-Line Options

```bash
# Basic options
python -m unified_test_harness.cli [OPTIONS]

Options:
  --source-root PATH          Source code root directory (default: current dir)
  --test-dir PATH             Test directory (default: source_root/tests)
  --output-dir PATH           Output directory (default: test_dir/harness_output)
  --project-type TYPE         Project structure: standard|src_layout|modules_layout|c_project|rust_project
  --language LANG             Language: python|c|rust (auto-detected)

# Actions
  --init                      Initialize vector database
  --coverage-only             Only run coverage analysis
  --generate-only             Only generate tests (skip coverage)
  
# LLM options
  --use-llm                   Enable AI-powered test generation
  --llm-provider PROVIDER     LLM provider: openai|anthropic (default: openai)
  --llm-api-key KEY           API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY)
  --llm-model MODEL           Model name (default: gpt-4)

# Generation options
  --modules MODULE [MODULE...] Focus on specific modules
  --batch-size SIZE           Batch size for processing (default: 50)
  --coverage-threshold FLOAT  Coverage threshold (default: 0.8)
  --no-vector-db              Disable vector database

# Examples
python -m unified_test_harness.cli --init
python -m unified_test_harness.cli --use-llm --llm-provider openai
python -m unified_test_harness.cli --coverage-only
python -m unified_test_harness.cli --modules auth database --use-llm
```

## 📁 Project Structure Support

The harness automatically detects your project structure:

### Standard Layout
```
project/
├── module1.py
├── module2.py
└── tests/
    ├── test_module1.py
    └── harness_output/
        ├── generated_tests/
        ├── test_vectors.json
        └── HARNESS_SUMMARY.md
```

### src/ Layout
```
project/
├── src/
│   ├── package/
│   │   ├── __init__.py
│   │   └── module.py
│   └── ...
└── tests/
    └── test_module.py
```

### C Project Layout
```
project/
├── src/
│   ├── module.c
│   └── module.h
├── tests/
│   └── test_module.c
└── Makefile
```

### Rust Project Layout
```
project/
├── Cargo.toml
├── src/
│   └── lib.rs
└── tests/
    └── integration_test.rs
```

## 🎨 How It Works

### 1. Coverage Analysis

The harness runs your existing test suite and analyzes code coverage:

- **Python**: Uses `pytest-cov` to generate coverage reports
- **C**: Uses `gcov`/`lcov` for coverage analysis
- **Rust**: Uses `cargo-tarpaulin` for coverage

### 2. Gap Identification

Identifies untested code by analyzing:
- Functions without test coverage
- Modules with low coverage percentages
- Edge cases and error paths not covered

### 3. Code Analysis

For each function, extracts:
- Function signatures and parameter types
- Return types
- Docstrings and documentation
- Exception types that might be raised
- Dependencies and imports

### 4. Test Generation

**Without LLM (Template-Based):**
- Finds similar code patterns in your codebase
- Adapts existing test templates
- Generates basic test structure

**With LLM (AI-Powered):**
- Analyzes function details and context
- Generates comprehensive test vectors
- Creates complete, executable test code
- Includes edge cases and error handling
- Refines output for accuracy

### 5. Output

Generates:
- **Test Vectors** (`test_vectors.json`): Structured test definitions
- **Test Code** (`generated_tests/`): Executable test files
- **Coverage Report** (`coverage.xml`, `htmlcov/`): Coverage analysis
- **Summary** (`HARNESS_SUMMARY.md`): Execution summary

## 📊 Example Output

### Generated Test (Python)

```python
"""
Test calculate_total function

Vector ID: calculator_calculate_total_unit
Type: unit
Priority: high
Coverage targets: calculate_total
"""
import pytest
from calculator import calculate_total

def test_calculator_calculate_total_unit():
    """
    Test calculate_total with normal inputs
    
    Coverage targets: calculate_total
    """
    # Prepare test inputs
    items = [{"price": 10.0}, {"price": 20.0}]
    discount = 0.1
    
    # Execute test
    result = calculate_total(items, discount)
    
    # Verify results
    assert result == 27.0  # (10 + 20) * 0.9
```

### Generated Test (Rust)

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate_total_unit() {
        // Prepare test inputs
        let items = vec![Item { price: 10.0 }, Item { price: 20.0 }];
        let discount = 0.1;
        
        // Execute test
        let result = calculate_total(&items, discount);
        
        // Verify results
        assert_eq!(result, 27.0);
    }
}
```

## 🔧 Configuration

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY=your_openai_key
export ANTHROPIC_API_KEY=your_anthropic_key

# Optional: Custom paths
export HARNESS_SOURCE_ROOT=/path/to/source
export HARNESS_TEST_DIR=/path/to/tests
```

### Configuration File

Create `harness_config.json`:

```json
{
  "source_root": ".",
  "test_dir": "tests",
  "output_dir": "tests/harness_output",
  "project_type": "standard",
  "language": "python",
  "llm_enabled": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "coverage_threshold": 0.8,
  "batch_size": 50,
  "use_vector_db": true
}
```

## 🛠️ Requirements

### Core Requirements

- **Python 3.8+**
- **pytest** (for Python test execution)
- **pytest-cov** (for Python coverage)
- **coverage** (for coverage analysis)

### Optional Dependencies

**Vector Database (Recommended):**
```bash
pip install chromadb
```

**LLM Providers:**
```bash
# OpenAI
pip install openai

# Anthropic
pip install anthropic
```

**C/C++ Support:**
```bash
# Ubuntu/Debian
sudo apt-get install gcc lcov

# macOS
brew install gcc lcov
```

**Rust Support:**
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install cargo-tarpaulin
cargo install cargo-tarpaulin
```

## 🐛 Troubleshooting

### "ChromaDB not available"

```bash
pip install chromadb
```

Or disable vector database:
```bash
python -m unified_test_harness.cli --no-vector-db
```

### "Coverage not found"

```bash
pip install pytest-cov coverage
```

### "LLM API error"

1. Check API key is set:
   ```bash
   export OPENAI_API_KEY=your_key_here
   ```

2. Verify API quota/limits

3. The harness will automatically fall back to template-based generation

### "No modules found"

1. Check `--source-root` points to correct directory
2. Verify project structure matches `--project-type`
3. Ensure source files match expected patterns (`.py`, `.c`, `.rs`)

### "Coverage command failed"

**Python:**
- Ensure pytest is installed: `pip install pytest pytest-cov`
- Check test directory exists and contains test files

**C:**
- Ensure project is compiled with `--coverage` flag
- Verify `lcov` is installed: `sudo apt-get install lcov`

**Rust:**
- Install cargo-tarpaulin: `cargo install cargo-tarpaulin`
- Ensure Cargo.toml exists in project root

### Generated tests have errors

1. Review generated tests in `tests/harness_output/generated_tests/`
2. Check import paths match your project structure
3. Verify function signatures match actual code
4. Use `--use-llm` for better quality generation

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - Quick start guide with examples
- **[EXAMPLES.md](EXAMPLES.md)** - Real-world usage examples
- **[C_RUST_SUPPORT.md](C_RUST_SUPPORT.md)** - C and Rust support details
- **[ML_GENERATION_ENHANCEMENTS.md](ML_GENERATION_ENHANCEMENTS.md)** - AI generation improvements
- **[unified_test_harness/README.md](unified_test_harness/README.md)** - Detailed API documentation

## 🤝 Contributing

We welcome contributions! When adding support for new frameworks:

1. Extend `FrameworkConfig` in `config.py`
2. Add test code generator in `llm_generator.py`
3. Update coverage command patterns
4. Add framework-specific fixtures in `conftest.py`
5. Update documentation

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This unified harness combines methodologies from:

- **KP14 Vector Database Testing Harness**: ChromaDB integration, template-based generation
- **GETMOVIN Test Harness**: LLM integration, pytest compatibility, test vector registry

## 💬 Support

For issues, questions, or contributions:

- Open an issue on GitHub
- Check the documentation in `unified_test_harness/README.md`
- Review `HARNESS_SUMMARY.md` after running for execution details

## 🎯 Best Practices

1. **Initialize First**: Always run `--init` before first use to set up the vector database
2. **Use LLM for Quality**: Enable `--use-llm` for production-quality test generation
3. **Review Generated Tests**: Always review and adjust generated tests before committing
4. **Regular Updates**: Re-run harness after significant code changes
5. **CI/CD Integration**: Integrate into your CI/CD pipeline for automated test generation
6. **Focus Modules**: Use `--modules` to focus on specific areas when needed

## 📈 Roadmap

- [ ] Support for more languages (Go, Java, TypeScript)
- [ ] Interactive test refinement
- [ ] Test execution and validation
- [ ] Integration with more test frameworks
- [ ] Web UI for test review and management
- [ ] Performance benchmarking

---

**Made with ❤️ for developers who want better test coverage with less effort.**
