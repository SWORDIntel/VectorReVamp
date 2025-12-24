# Usage Examples

Real-world examples of using Unified Test Harness in different scenarios.

## Example 1: Python Web Application

### Project Structure
```
webapp/
├── src/
│   └── app/
│       ├── __init__.py
│       ├── auth.py
│       ├── database.py
│       └── api.py
└── tests/
    └── test_auth.py
```

### Setup and Run

```bash
cd webapp

# Initialize
python -m unified_test_harness.cli --project-type src_layout --init

# Generate tests with AI
export OPENAI_API_KEY=your_key
python -m unified_test_harness.cli --project-type src_layout --use-llm

# Focus on authentication module
python -m unified_test_harness.cli --project-type src_layout --modules auth --use-llm
```

### Generated Test Example

```python
"""
Test authenticate_user function

Vector ID: auth_authenticate_user_unit
Type: unit
Priority: high
"""
import pytest
from app.auth import authenticate_user

def test_auth_authenticate_user_unit():
    """Test authenticate_user with valid credentials"""
    # Prepare test inputs
    username = "testuser"
    password = "testpass123"
    
    # Execute test
    result = authenticate_user(username, password)
    
    # Verify results
    assert result is not None
    assert result.username == username
```

## Example 2: C Library Project

### Project Structure
```
clib/
├── src/
│   ├── math_utils.c
│   └── math_utils.h
├── tests/
│   └── test_math_utils.c
└── Makefile
```

### Makefile Setup

```makefile
CC = gcc
CFLAGS = -Wall -Wextra --coverage
LDFLAGS = --coverage

test: test_math_utils
	./test_math_utils

coverage: test
	lcov --capture --directory . --output-file coverage.info
	genhtml coverage.info --output-directory htmlcov
```

### Setup and Run

```bash
cd clib

# Compile with coverage
make clean
make CFLAGS="--coverage" test

# Initialize harness
python -m unified_test_harness.cli --language c --init

# Generate tests
python -m unified_test_harness.cli --language c --use-llm
```

### Generated Test Example

```c
/*
 * Test add_numbers function
 */
#include "unity.h"
#include "math_utils.h"

void setUp(void) {
    // Setup
}

void tearDown(void) {
    // Teardown
}

void test_add_numbers_normal(void) {
    // Execute test
    int result = add_numbers(10, 20);
    
    // Verify results
    TEST_ASSERT_EQUAL(30, result);
}
```

## Example 3: Rust CLI Application

### Project Structure
```
rust_cli/
├── Cargo.toml
├── src/
│   ├── main.rs
│   └── lib.rs
└── tests/
    └── integration_test.rs
```

### Cargo.toml
```toml
[package]
name = "rust_cli"
version = "0.1.0"

[dev-dependencies]
```

### Setup and Run

```bash
cd rust_cli

# Install cargo-tarpaulin
cargo install cargo-tarpaulin

# Initialize harness
python -m unified_test_harness.cli --language rust --init

# Generate tests with AI
export OPENAI_API_KEY=your_key
python -m unified_test_harness.cli --language rust --use-llm
```

### Generated Test Example

```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_process_data_normal() {
        // Prepare test inputs
        let data = vec![1, 2, 3, 4, 5];
        
        // Execute test
        let result = process_data(&data);
        
        // Verify results
        assert_eq!(result, 15);
    }
}
```

## Example 4: Python API Integration

### Using the Python API

```python
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

# Configure for your project
config = HarnessConfig.create_for_project(
    source_root=Path("/path/to/project"),
    project_type="src_layout"
)

# Enable AI generation
config.llm_enabled = True
config.llm_provider = "openai"
config.llm_api_key = "sk-..."

# Customize settings
config.coverage_threshold = 0.85
config.batch_size = 25
config.max_tests_per_module = 50

# Create runner
runner = TestHarnessRunner(config)

# Initialize
runner.initialize()

# Run coverage
coverage = runner.run_coverage_analysis()
print(f"Coverage: {coverage['coverage_percentage']:.2f}%")

# Find gaps
gaps = runner.identify_gaps()
print(f"Found {len(gaps)} gaps")

# Generate tests
vectors = runner.generate_test_vectors(gaps, use_llm=True)
print(f"Generated {len(vectors)} vectors")

# Save everything
runner.save_generated_tests()
runner.save_results()

# Access results
print(f"Tests saved to: {config.output_dir / 'generated_tests'}")
```

## Example 5: CI/CD Integration

### GitHub Actions

```yaml
name: Generate Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  generate-tests:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install -e .
        pip install chromadb openai
    
    - name: Initialize harness
      run: |
        python -m unified_test_harness.cli --init
    
    - name: Generate tests
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
      run: |
        python -m unified_test_harness.cli --use-llm
    
    - name: Run generated tests
      run: |
        pytest tests/harness_output/generated_tests/ \
          --cov=. \
          --cov-report=xml \
          --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./tests/harness_output/coverage.xml
```

### GitLab CI

```yaml
generate_tests:
  image: python:3.9
  before_script:
    - pip install -r requirements.txt
    - pip install -e .
    - pip install chromadb openai
  script:
    - python -m unified_test_harness.cli --init
    - python -m unified_test_harness.cli --use-llm
    - pytest tests/harness_output/generated_tests/ --cov=. --cov-report=xml
  artifacts:
    paths:
      - tests/harness_output/
    reports:
      coverage_report:
        coverage_format: cobertura
        path: tests/harness_output/coverage.xml
```

## Example 6: Focused Module Testing

### Test Specific Modules

```bash
# Test only authentication and database modules
python -m unified_test_harness.cli \
  --modules auth database \
  --use-llm \
  --batch-size 10

# Test with custom coverage threshold
python -m unified_test_harness.cli \
  --modules api \
  --coverage-threshold 0.9 \
  --use-llm
```

### Python API - Focused Testing

```python
from unified_test_harness import HarnessConfig, TestHarnessRunner

config = HarnessConfig.create_for_project(Path("."), "standard")
config.llm_enabled = True

runner = TestHarnessRunner(config)

# Focus on specific modules
results = runner.run_full_harness(
    focus_modules=["auth", "database", "api"],
    use_llm=True
)

print(f"Generated {len(results['generated_vectors'])} vectors")
```

## Example 7: Custom Configuration

### Configuration File

Create `harness_config.json`:

```json
{
  "source_root": ".",
  "test_dir": "tests",
  "output_dir": "tests/harness_output",
  "project_type": "src_layout",
  "language": "python",
  "llm_enabled": true,
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "llm_api_key": "sk-...",
  "coverage_threshold": 0.85,
  "coverage_minimum": 0.0,
  "batch_size": 50,
  "max_tests_per_module": 100,
  "use_vector_db": true,
  "vector_db_path": ".harness_db",
  "generate_unit_tests": true,
  "generate_integration_tests": true,
  "generate_edge_cases": true,
  "generate_error_tests": true,
  "save_generated_tests": true,
  "test_output_format": "pytest"
}
```

### Using Configuration File

```python
import json
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

# Load configuration
with open("harness_config.json") as f:
    config_data = json.load(f)

# Create config from dict
config = HarnessConfig.from_dict(config_data)

# Create runner
runner = TestHarnessRunner(config)

# Run harness
results = runner.run_full_harness()
```

## Example 8: Batch Processing

### Process Multiple Projects

```bash
#!/bin/bash

projects=("project1" "project2" "project3")

for project in "${projects[@]}"; do
    echo "Processing $project..."
    cd "$project"
    
    python -m unified_test_harness.cli --init
    python -m unified_test_harness.cli --use-llm
    
    cd ..
done
```

### Python Script for Batch Processing

```python
from pathlib import Path
from unified_test_harness import HarnessConfig, TestHarnessRunner

projects = ["project1", "project2", "project3"]

for project_name in projects:
    project_path = Path(project_name)
    
    print(f"Processing {project_name}...")
    
    config = HarnessConfig.create_for_project(project_path, "standard")
    config.llm_enabled = True
    config.llm_api_key = "your-key"
    
    runner = TestHarnessRunner(config)
    results = runner.run_full_harness()
    
    print(f"  Coverage: {results['coverage_report']['coverage_percentage']:.2f}%")
    print(f"  Generated: {len(results['generated_vectors'])} vectors")
```

## Example 9: Custom Test Generation

### Using Test Vectors Directly

```python
from unified_test_harness import TestVector, TestVectorType, TestPriority
from unified_test_harness.llm_generator import LLMTestGenerator

# Create custom test vector
vector = TestVector(
    vector_id="custom_test_1",
    name="Custom test for my function",
    description="Tests specific behavior",
    module_name="mymodule",
    vector_type=TestVectorType.UNIT,
    priority=TestPriority.HIGH,
    inputs={"x": 10, "y": 20},
    expected_outputs={"result": 30},
    coverage_targets=["add"],
    framework_config={"test_framework": "pytest"}
)

# Generate test code
generator = LLMTestGenerator(config, coverage_analyzer, code_embedder)
test_code = generator.generate_test_code(vector)

print(test_code)
```

## Example 10: Coverage Analysis Only

### Just Check Coverage

```bash
# Run coverage analysis without generating tests
python -m unified_test_harness.cli --coverage-only

# View coverage report
open tests/harness_output/htmlcov/index.html

# Check coverage percentage
cat tests/harness_output/HARNESS_SUMMARY.md | grep Coverage
```

### Python API - Coverage Only

```python
from unified_test_harness import HarnessConfig, TestHarnessRunner

config = HarnessConfig.create_for_project(Path("."), "standard")
runner = TestHarnessRunner(config)

# Run coverage analysis
coverage = runner.run_coverage_analysis()

# Print results
print(f"Coverage: {coverage['coverage_percentage']:.2f}%")
print(f"Total functions: {coverage['total_functions']}")
print(f"Covered functions: {coverage['covered_functions']}")

# Identify gaps
gaps = runner.identify_gaps()
print(f"Found {len(gaps)} coverage gaps")

for gap in gaps[:5]:
    print(f"  - {gap['module']}: {len(gap['uncovered_functions'])} uncovered functions")
```

---

These examples cover common use cases. For more details, see the [README.md](README.md) and [QUICKSTART.md](QUICKSTART.md).
