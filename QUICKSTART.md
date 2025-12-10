# Quick Start Guide - Unified Test Harness

## Installation

```bash
# Clone or download the unified_test_harness directory
cd unified_test_harness

# Install dependencies
pip install -r ../requirements.txt

# Or install with extras
pip install -e ".[all]"  # Includes vector DB and LLM support
```

## Basic Usage

### Step 1: Initialize

```bash
# For your project (from project root)
python -m unified_test_harness.cli --init

# Or specify project type
python -m unified_test_harness.cli --project-type src_layout --init
```

### Step 2: Run Coverage Analysis

```bash
python -m unified_test_harness.cli --coverage-only
```

### Step 3: Generate Tests

```bash
# Without LLM (uses templates and similarity)
python -m unified_test_harness.cli

# With LLM (requires API key)
export OPENAI_API_KEY=your_key_here
python -m unified_test_harness.cli --use-llm
```

## Project Structure Detection

The harness automatically detects your project structure:

### Standard Layout
```
project/
├── module1.py
├── module2.py
└── tests/
    └── test_module1.py
```

### src/ Layout
```
project/
├── src/
│   ├── module1.py
│   └── module2.py
└── tests/
    └── test_module1.py
```

### modules/ Layout
```
project/
├── modules/
│   ├── module1.py
│   └── module2.py
└── tests/
    └── test_module1.py
```

## Configuration Examples

### Python API

```python
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

# Create config
config = HarnessConfig.create_for_project(Path("."), "standard")

# Enable LLM
config.llm_enabled = True
config.llm_provider = "openai"
config.llm_api_key = "your-key-here"

# Create runner
runner = TestHarnessRunner(config)

# Run harness
results = runner.run_full_harness()
```

### CLI Configuration

Create a config file `harness_config.json`:

```json
{
  "source_root": ".",
  "test_dir": "tests",
  "output_dir": "tests/harness_output",
  "project_type": "standard",
  "llm_enabled": true,
  "llm_provider": "openai",
  "batch_size": 50,
  "coverage_threshold": 0.8
}
```

## Common Workflows

### Daily Development

```bash
# After code changes
python -m unified_test_harness.cli --coverage-only

# Generate tests for new modules
python -m unified_test_harness.cli --modules new_module1 new_module2
```

### CI/CD Integration

```bash
# In CI pipeline
python -m unified_test_harness.cli --init
python -m unified_test_harness.cli --use-llm
pytest tests/harness_output/generated_tests/
```

### Focused Testing

```bash
# Test specific modules only
python -m unified_test_harness.cli --modules router orchestrator

# Larger batch size for bulk generation
python -m unified_test_harness.cli --batch-size 100
```

## Output Files

After running, check:

- `tests/harness_output/HARNESS_SUMMARY.md` - Summary report
- `tests/harness_output/test_vectors.json` - Generated test vectors
- `tests/harness_output/generated_tests/` - Executable test code
- `tests/harness_output/coverage.xml` - Coverage report

## Troubleshooting

### "ChromaDB not available"
```bash
pip install chromadb
```

### "Coverage not found"
```bash
pip install pytest-cov coverage
```

### "LLM API error"
- Check API key is set: `export OPENAI_API_KEY=your_key`
- Verify quota/limits
- Harness will fall back to template-based generation

### "No modules found"
- Check `--source-root` points to correct directory
- Verify project structure matches `--project-type`
- Check source patterns in config

## Next Steps

1. Review generated tests in `tests/harness_output/generated_tests/`
2. Run generated tests: `pytest tests/harness_output/generated_tests/`
3. Review coverage report: `open tests/harness_output/htmlcov/index.html`
4. Integrate into CI/CD pipeline
5. Customize configuration for your project

## Support

For issues or questions:
- Check `README.md` for detailed documentation
- Review `HARNESS_SUMMARY.md` for execution details
- Check generated test code for adaptation patterns
