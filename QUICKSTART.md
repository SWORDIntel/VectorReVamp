# Quick Start Guide

Get up and running with Unified Test Harness in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Your project codebase (Python, C, or Rust)

## Installation

### Step 1: Install the Package

```bash
# Clone or download the repository
git clone <repository-url>
cd unified-test-harness

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .

# Or install with all optional features
pip install -e ".[all]"
```

### Step 2: Install Optional Dependencies

**For AI-powered generation (recommended):**
```bash
pip install openai  # or anthropic
```

**For vector database (recommended for better results):**
```bash
pip install chromadb
```

**For C projects:**
```bash
# Ubuntu/Debian
sudo apt-get install gcc lcov

# macOS
brew install gcc lcov
```

**For Rust projects:**
```bash
cargo install cargo-tarpaulin
```

## Your First Run

### Python Project Example

```bash
# 1. Navigate to your project
cd /path/to/your/python/project

# 2. Initialize the harness
python -m unified_test_harness.cli --init

# 3. Run coverage analysis
python -m unified_test_harness.cli --coverage-only

# 4. Generate tests (without AI)
python -m unified_test_harness.cli

# 5. Generate tests with AI (better quality)
export OPENAI_API_KEY=your_api_key_here
python -m unified_test_harness.cli --use-llm
```

### What You'll See

```
[*] Embedding codebase from /path/to/project...
[*] Found 15 Python files
[*] Extracted 45 code segments
[+] Codebase embedding complete!

[*] Running coverage analysis...
[+] Coverage: 65.23%
[+] Total functions: 120
[+] Covered functions: 78

[*] Identifying coverage gaps...
[+] Found 8 coverage gaps

[*] Generating test vectors...
[+] Generated 24 test vectors

[+] Saved 24 tests to tests/harness_output/generated_tests/
```

### Review Generated Tests

```bash
# View generated tests
ls tests/harness_output/generated_tests/

# Run generated tests
pytest tests/harness_output/generated_tests/

# View coverage report
open tests/harness_output/htmlcov/index.html

# Read summary
cat tests/harness_output/HARNESS_SUMMARY.md
```

## Common Workflows

### Workflow 1: New Feature Development

```bash
# After writing new code
python -m unified_test_harness.cli --coverage-only

# Generate tests for new modules
python -m unified_test_harness.cli --modules new_feature --use-llm

# Review and run tests
pytest tests/harness_output/generated_tests/test_new_feature_generated.py
```

### Workflow 2: Improving Coverage

```bash
# Check current coverage
python -m unified_test_harness.cli --coverage-only

# Generate tests for low-coverage modules
python -m unified_test_harness.cli --modules low_coverage_module --use-llm

# Review generated tests
cat tests/harness_output/generated_tests/test_low_coverage_module_generated.py
```

### Workflow 3: CI/CD Integration

```bash
# In your CI script
python -m unified_test_harness.cli --init
python -m unified_test_harness.cli --use-llm
pytest tests/harness_output/generated_tests/ --cov=. --cov-report=xml
```

### Workflow 4: Focused Testing

```bash
# Test specific modules only
python -m unified_test_harness.cli --modules auth database api --use-llm

# Larger batch for bulk generation
python -m unified_test_harness.cli --batch-size 100 --use-llm
```

## Project Structure Examples

### Standard Python Project

```
my_project/
├── calculator.py
├── database.py
└── tests/
    ├── test_calculator.py
    └── harness_output/
        ├── generated_tests/
        │   ├── test_calculator_generated.py
        │   └── test_database_generated.py
        ├── test_vectors.json
        └── HARNESS_SUMMARY.md
```

**Command:**
```bash
python -m unified_test_harness.cli --init
python -m unified_test_harness.cli
```

### src/ Layout Python Project

```
my_project/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       └── module.py
└── tests/
    └── test_module.py
```

**Command:**
```bash
python -m unified_test_harness.cli --project-type src_layout --init
python -m unified_test_harness.cli --project-type src_layout
```

### C Project

```
my_c_project/
├── src/
│   ├── calculator.c
│   └── calculator.h
├── tests/
│   └── test_calculator.c
└── Makefile
```

**Command:**
```bash
python -m unified_test_harness.cli --language c --init
python -m unified_test_harness.cli --language c
```

### Rust Project

```
my_rust_project/
├── Cargo.toml
├── src/
│   └── lib.rs
└── tests/
    └── integration_test.rs
```

**Command:**
```bash
python -m unified_test_harness.cli --language rust --init
python -m unified_test_harness.cli --language rust
```

## Using the Python API

### Basic Example

```python
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

# Create configuration
config = HarnessConfig.create_for_project(Path("."), "standard")

# Enable AI generation
config.llm_enabled = True
config.llm_provider = "openai"
config.llm_api_key = "your-api-key"

# Create runner
runner = TestHarnessRunner(config)

# Run full harness
results = runner.run_full_harness()

# Check results
print(f"Coverage: {results['coverage_report']['coverage_percentage']:.2f}%")
print(f"Generated: {len(results['generated_vectors'])} test vectors")
```

### Advanced Example

```python
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

# Configure
config = HarnessConfig.create_for_project(Path("."), "src_layout")

# Customize settings
config.llm_enabled = True
config.llm_provider = "openai"
config.llm_model = "gpt-4"
config.coverage_threshold = 0.8
config.batch_size = 50
config.use_vector_db = True

# Create runner
runner = TestHarnessRunner(config)

# Initialize vector database
runner.initialize()

# Run coverage analysis
coverage = runner.run_coverage_analysis()
print(f"Current coverage: {coverage['coverage_percentage']:.2f}%")

# Identify gaps
gaps = runner.identify_gaps()
print(f"Found {len(gaps)} coverage gaps")

# Generate tests for specific modules
vectors = runner.generate_test_vectors(gaps[:5], use_llm=True)
print(f"Generated {len(vectors)} test vectors")

# Save results
runner.save_generated_tests()
runner.save_results()
```

## Configuration Options

### Command Line

```bash
# Basic options
--source-root PATH          # Source code directory
--test-dir PATH             # Test directory
--output-dir PATH           # Output directory
--project-type TYPE         # Project structure type
--language LANG             # Programming language

# Actions
--init                      # Initialize vector database
--coverage-only             # Only run coverage
--generate-only             # Only generate tests

# LLM options
--use-llm                   # Enable AI generation
--llm-provider PROVIDER     # openai or anthropic
--llm-api-key KEY           # API key
--llm-model MODEL           # Model name

# Generation
--modules MODULE [MODULE...] # Focus on modules
--batch-size SIZE           # Batch size
--coverage-threshold FLOAT  # Coverage threshold
```

### Environment Variables

```bash
export OPENAI_API_KEY=your_key
export ANTHROPIC_API_KEY=your_key
export HARNESS_SOURCE_ROOT=/path/to/source
export HARNESS_TEST_DIR=/path/to/tests
```

## Understanding the Output

### Generated Files

After running, you'll find:

1. **`tests/harness_output/generated_tests/`**
   - Executable test files
   - Ready to run with pytest/unittest/cargo test

2. **`tests/harness_output/test_vectors.json`**
   - Structured test definitions
   - Can be used for custom test generation

3. **`tests/harness_output/HARNESS_SUMMARY.md`**
   - Execution summary
   - Coverage statistics
   - Generated test overview

4. **`tests/harness_output/coverage.xml`**
   - Coverage data in XML format
   - Compatible with CI/CD tools

5. **`tests/harness_output/htmlcov/`**
   - HTML coverage report
   - Visual coverage analysis

### Test Vector Structure

Each test vector includes:

```json
{
  "vector_id": "module_function_test_type",
  "name": "Test function_name with normal inputs",
  "description": "Validates function behavior with standard inputs",
  "module_name": "calculator",
  "vector_type": "unit",
  "priority": "high",
  "inputs": {"x": 10, "y": 20},
  "expected_outputs": {"result": 30},
  "expected_errors": [],
  "coverage_targets": ["add"],
  "coverage_minimum": 0.8,
  "tags": ["unit", "basic"]
}
```

## Troubleshooting

### Issue: "ChromaDB not available"

**Solution:**
```bash
pip install chromadb
```

Or disable vector database:
```bash
python -m unified_test_harness.cli --no-vector-db
```

### Issue: "Coverage command failed"

**For Python:**
```bash
pip install pytest pytest-cov coverage
```

**For C:**
- Ensure project compiled with `--coverage` flag
- Install lcov: `sudo apt-get install lcov`

**For Rust:**
```bash
cargo install cargo-tarpaulin
```

### Issue: "LLM API error"

**Solutions:**
1. Check API key: `echo $OPENAI_API_KEY`
2. Verify quota/limits in your API account
3. Harness will fall back to template-based generation automatically

### Issue: "No modules found"

**Solutions:**
1. Check `--source-root` points to correct directory
2. Verify project structure matches `--project-type`
3. Ensure source files have correct extensions (`.py`, `.c`, `.rs`)

### Issue: Generated tests have import errors

**Solutions:**
1. Review import paths in generated tests
2. Adjust `import_prefix` in configuration
3. Use `--project-type` to match your structure

## Next Steps

1. ✅ **Review Generated Tests**: Check `tests/harness_output/generated_tests/`
2. ✅ **Run Tests**: `pytest tests/harness_output/generated_tests/`
3. ✅ **Review Coverage**: Open `tests/harness_output/htmlcov/index.html`
4. ✅ **Integrate CI/CD**: Add to your pipeline
5. ✅ **Customize**: Adjust configuration for your project

## Getting Help

- 📖 Read the full [README.md](README.md)
- 📚 Check [ML_GENERATION_ENHANCEMENTS.md](ML_GENERATION_ENHANCEMENTS.md) for AI features
- 🔧 Review `tests/harness_output/HARNESS_SUMMARY.md` after running
- 💬 Open an issue on GitHub

## Tips for Best Results

1. **Use AI Generation**: Enable `--use-llm` for better quality
2. **Initialize First**: Always run `--init` before first use
3. **Review Tests**: Always review generated tests before committing
4. **Focus Modules**: Use `--modules` for targeted generation
5. **Regular Updates**: Re-run after significant code changes
6. **CI Integration**: Add to your CI/CD for automated generation

---

**Ready to generate better tests? Run `python -m unified_test_harness.cli --init` and get started!**
