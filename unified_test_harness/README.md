# Unified Test Harness

A framework-agnostic, vector database-driven testing harness that combines coverage analysis, LLM-powered test generation, and template-based test creation. Works with any Python project and testing framework.

## Features

- **Framework Agnostic**: Works with pytest, unittest, nose, and custom test frameworks
- **Vector Database Integration**: Uses ChromaDB for semantic code similarity search
- **LLM-Powered Generation**: Supports OpenAI and Anthropic for intelligent test generation
- **Coverage-Driven**: Identifies gaps and prioritizes test generation
- **Template-Based**: Learns from existing tests and adapts them to new code
- **Configurable**: Supports different project structures (standard, src/, modules/)
- **CI/CD Ready**: Designed for integration into automated workflows

## Installation

```bash
# Install from source
pip install -e .

# Or install dependencies manually
pip install chromadb coverage pytest
```

## Quick Start

### 1. Initialize the Harness

```bash
# For standard project structure
unified-test-harness --init

# For src/ layout
unified-test-harness --project-type src_layout --init

# For modules/ layout
unified-test-harness --project-type modules_layout --init
```

### 2. Run Full Harness

```bash
# Basic run (no LLM)
unified-test-harness

# With LLM generation
unified-test-harness --use-llm --llm-provider openai

# Focus on specific modules
unified-test-harness --modules module1 module2
```

### 3. Coverage Analysis Only

```bash
unified-test-harness --coverage-only
```

### 4. Generate Tests Only

```bash
unified-test-harness --generate-only
```

## Configuration

The harness automatically detects your project structure, but you can customize it:

### Project Types

- **standard**: Flat structure with tests/ directory
- **src_layout**: Source code in src/, tests in tests/
- **modules_layout**: Source code in modules/, tests in tests/

### Environment Variables

```bash
# LLM API Keys (if not provided via --llm-api-key)
export OPENAI_API_KEY=your_key_here
export ANTHROPIC_API_KEY=your_key_here
```

## Usage Examples

### Python API

```python
from unified_test_harness import HarnessConfig, TestHarnessRunner
from pathlib import Path

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

### CLI Examples

```bash
# Initialize and run
unified-test-harness --init
unified-test-harness --use-llm

# Custom paths
unified-test-harness \
    --source-root /path/to/project \
    --test-dir /path/to/tests \
    --output-dir /path/to/output

# Batch processing
unified-test-harness --batch-size 100

# Specific modules
unified-test-harness --modules router orchestrator plugin_manager
```

## Architecture

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
└── cli.py                   # Command-line interface
```

## Workflow

1. **Initialization**: Embeds codebase and existing tests into vector database
2. **Coverage Analysis**: Runs test suite with coverage tracking
3. **Gap Identification**: Identifies uncovered functions and modules
4. **Test Generation**: Generates test vectors using:
   - Vector similarity search (finds similar code patterns)
   - LLM generation (if enabled)
   - Template adaptation (adapts existing test patterns)
5. **Test Code Generation**: Converts vectors to executable test code
6. **Results**: Saves test vectors, generated tests, and reports

## Output Structure

```
tests/harness_output/
├── vector_db/               # ChromaDB vector database
├── coverage.xml             # Coverage XML report
├── htmlcov/                 # HTML coverage report
├── test_results.json        # Test execution results
├── test_vectors.json        # Generated test vectors
├── HARNESS_SUMMARY.md       # Summary report
└── generated_tests/         # Generated test code
    ├── test_module1_generated.py
    └── test_module2_generated.py
```

## Integration with CI/CD

### GitHub Actions

```yaml
name: Test Generation

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  generate-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install unified-test-harness
      - run: unified-test-harness --init
      - run: unified-test-harness --use-llm
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      - run: pytest tests/harness_output/generated_tests/
```

### GitLab CI

```yaml
test_generation:
  script:
    - pip install unified-test-harness
    - unified-test-harness --init
    - unified-test-harness --use-llm
  only:
    - schedules
```

## Best Practices

1. **Initialize Once**: Run `--init` once, then use regular runs for updates
2. **Regular Analysis**: Run coverage analysis after significant code changes
3. **Review Generated Tests**: Always review LLM-generated tests before committing
4. **Batch Sizes**: Start with 20-50, increase as needed
5. **Coverage Thresholds**: Maintain 80%+ coverage
6. **Version Control**: Commit generated tests after review

## Troubleshooting

### ChromaDB Not Available

```bash
pip install chromadb
```

### Coverage Module Not Available

```bash
pip install coverage pytest-cov
```

### LLM API Errors

- Check API key is set correctly
- Verify API quota/limits
- Harness falls back to basic generation if LLM unavailable

### Vector Database Locked

```bash
rm -rf tests/harness_output/vector_db/chroma.sqlite3
unified-test-harness --init
```

## Requirements

- Python 3.8+
- pytest (for test execution)
- pytest-cov (for coverage)
- chromadb (for vector database, optional)
- openai or anthropic (for LLM generation, optional)

## License

Same as main project (for authorized security testing only).

## Contributing

When adding support for new frameworks:

1. Extend `FrameworkConfig` in `config.py`
2. Add test code generator in `llm_generator.py`
3. Update coverage command patterns
4. Add framework-specific fixtures in `conftest.py`

## Credits

Based on methodologies from:
- KP14 Vector Database Testing Harness
- GETMOVIN Test Harness

Unified and generalized for any Python project.
