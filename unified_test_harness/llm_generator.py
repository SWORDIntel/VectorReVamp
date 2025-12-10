"""
LLM-Powered Test Generator

Uses LLM agents to generate test vectors based on coverage gaps and code analysis.
Framework-agnostic implementation.
Supports Python, C, and Rust test generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast

from .test_vector import TestVector, TestVectorType, TestPriority
from .language_parser import LanguageParser, Language


class LLMTestGenerator:
    """LLM-powered test generator"""
    
    def __init__(self, config, coverage_analyzer, code_embedder):
        """
        Initialize LLM test generator
        
        Args:
            config: HarnessConfig instance
            coverage_analyzer: CoverageAnalyzer instance
            code_embedder: CodeEmbedder instance
        """
        self.config = config
        self.coverage_analyzer = coverage_analyzer
        self.code_embedder = code_embedder
        self.generated_vectors: List[TestVector] = []
        self.llm_client = None
        self.language_parser = LanguageParser()
        
        if config.llm_enabled:
            self._init_llm_client()
    
    def _init_llm_client(self):
        """Initialize LLM client based on configuration"""
        provider = self.config.llm_provider.lower()
        
        if provider == "openai":
            try:
                import openai
                if self.config.llm_api_key:
                    self.llm_client = openai.OpenAI(api_key=self.config.llm_api_key)
                else:
                    # Try environment variable
                    import os
                    api_key = os.getenv("OPENAI_API_KEY")
                    if api_key:
                        self.llm_client = openai.OpenAI(api_key=api_key)
            except ImportError:
                print("[!] OpenAI library not installed. Install with: pip install openai")
        elif provider == "anthropic":
            try:
                import anthropic
                if self.config.llm_api_key:
                    self.llm_client = anthropic.Anthropic(api_key=self.config.llm_api_key)
                else:
                    import os
                    api_key = os.getenv("ANTHROPIC_API_KEY")
                    if api_key:
                        self.llm_client = anthropic.Anthropic(api_key=api_key)
            except ImportError:
                print("[!] Anthropic library not installed. Install with: pip install anthropic")
        else:
            print(f"[!] LLM provider '{provider}' not supported")
    
    def generate_test_vector_prompt(self, module_name: str, uncovered_functions: List[str], 
                                   module_code: str, file_path: str) -> str:
        """Generate prompt for LLM to create test vectors"""
        import_prefix = self.config.framework.import_prefix
        
        prompt = f"""Generate comprehensive test vectors for the following Python module.

Module: {module_name}
File Path: {file_path}
Import Prefix: {import_prefix}
Uncovered Functions: {', '.join(uncovered_functions)}

Module Code:
```python
{module_code}
```

Generate test vectors following these requirements:
1. Create test vectors for each uncovered function
2. Include unit tests, integration tests, and edge cases
3. Define inputs, expected outputs, and error cases
4. Specify coverage targets (functions/classes to cover)
5. Set appropriate priority levels (critical, high, medium, low)
6. Include preconditions and postconditions
7. Use the test framework: {self.config.framework.test_framework}
8. Use import prefix: {import_prefix}

Return JSON format:
{{
    "vectors": [
        {{
            "vector_id": "unique_id",
            "name": "Test name",
            "description": "What this test validates",
            "module_name": "{module_name}",
            "vector_type": "unit|integration|end_to_end|edge_case|error_handling",
            "priority": "critical|high|medium|low",
            "inputs": {{"param1": "value1"}},
            "expected_outputs": {{"result": "expected"}},
            "expected_errors": ["ErrorType"],
            "coverage_targets": ["function_name", "ClassName.method_name"],
            "coverage_minimum": 0.8,
            "preconditions": ["condition1"],
            "postconditions": ["condition2"],
            "tags": ["tag1", "tag2"]
        }}
    ]
}}
"""
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM API"""
        if not self.llm_client:
            return ""
        
        provider = self.config.llm_provider.lower()
        
        try:
            if provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.config.llm_model,
                    messages=[
                        {"role": "system", "content": "You are a test generation expert. Generate comprehensive test vectors in JSON format."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7
                )
                return response.choices[0].message.content
            elif provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.config.llm_model,
                    max_tokens=4000,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
        except Exception as e:
            print(f"[!] LLM API error: {e}")
            return ""
        
        return ""
    
    def parse_llm_response(self, response: str) -> List[TestVector]:
        """Parse LLM response and create test vectors"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start == -1 or json_end == 0:
                return []
            
            json_str = response[json_start:json_end]
            data = json.loads(json_str)
            
            vectors = []
            for vec_data in data.get('vectors', []):
                try:
                    vector = TestVector(
                        vector_id=vec_data['vector_id'],
                        name=vec_data['name'],
                        description=vec_data['description'],
                        module_name=vec_data['module_name'],
                        vector_type=TestVectorType(vec_data['vector_type']),
                        priority=TestPriority(vec_data['priority']),
                        inputs=vec_data.get('inputs', {}),
                        expected_outputs=vec_data.get('expected_outputs', {}),
                        expected_errors=vec_data.get('expected_errors', []),
                        coverage_targets=vec_data.get('coverage_targets', []),
                        coverage_minimum=vec_data.get('coverage_minimum', 0.8),
                        preconditions=vec_data.get('preconditions', []),
                        postconditions=vec_data.get('postconditions', []),
                        tags=vec_data.get('tags', []),
                        framework_config={
                            'import_prefix': self.config.framework.import_prefix,
                            'test_framework': self.config.framework.test_framework,
                        }
                    )
                    vectors.append(vector)
                except Exception as e:
                    print(f"[!] Error parsing vector: {e}")
                    continue
            
            return vectors
        except Exception as e:
            print(f"[!] Error parsing LLM response: {e}")
            return []
    
    def generate_vectors_for_module(self, module_name: str, priority: str = "medium") -> List[TestVector]:
        """Generate test vectors for a specific module"""
        # Find module file
        module_path = self.coverage_analyzer._find_module_file(module_name)
        if not module_path or not module_path.exists():
            return []
        
        # Read module code
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                module_code = f.read()
        except Exception as e:
            print(f"[!] Error reading module {module_name}: {e}")
            return []
        
        # Get uncovered functions
        uncovered = self.coverage_analyzer.get_uncovered_functions(module_name)
        
        if not uncovered:
            return []
        
        # Limit number of functions to process
        uncovered = uncovered[:self.config.max_tests_per_module]
        
        # Generate prompt
        file_path = str(module_path.relative_to(self.config.source_root))
        prompt = self.generate_test_vector_prompt(module_name, uncovered, module_code, file_path)
        
        # Call LLM (if available)
        if self.config.llm_enabled and self.llm_client:
            response = self._call_llm(prompt)
            if response:
                vectors = self.parse_llm_response(response)
            else:
                vectors = self._generate_basic_vectors(module_name, uncovered, priority)
        else:
            # Fallback: Generate basic vectors without LLM
            vectors = self._generate_basic_vectors(module_name, uncovered, priority)
        
        self.generated_vectors.extend(vectors)
        return vectors
    
    def _generate_basic_vectors(self, module_name: str, uncovered_functions: List[str], 
                               priority: str = "medium") -> List[TestVector]:
        """Generate basic test vectors without LLM"""
        vectors = []
        
        for func_name in uncovered_functions:
            # Determine vector type based on function name patterns
            vector_type = TestVectorType.UNIT
            if 'integration' in func_name.lower() or 'integration' in module_name.lower():
                vector_type = TestVectorType.INTEGRATION
            elif 'error' in func_name.lower() or 'exception' in func_name.lower():
                vector_type = TestVectorType.ERROR_HANDLING
            
            vector = TestVector(
                vector_id=f"{module_name}_{func_name}_basic",
                name=f"Test {func_name}",
                description=f"Basic test for {func_name} function in {module_name}",
                module_name=module_name,
                vector_type=vector_type,
                priority=TestPriority(priority),
                coverage_targets=[func_name],
                coverage_minimum=self.config.coverage_threshold,
                tags=["auto_generated", "basic"],
                framework_config={
                    'import_prefix': self.config.framework.import_prefix,
                    'test_framework': self.config.framework.test_framework,
                }
            )
            vectors.append(vector)
        
        return vectors
    
    def generate_test_code(self, vector: TestVector) -> str:
        """Generate actual test code from a test vector"""
        # Detect language from module file
        module_path = self.coverage_analyzer._find_module_file(vector.module_name)
        if module_path:
            language = self.language_parser.detect_language(module_path)
        else:
            language = Language.PYTHON  # Default
        
        import_prefix = vector.framework_config.get('import_prefix', '')
        test_framework = vector.framework_config.get('test_framework', 'pytest')
        
        if language == Language.PYTHON:
            if test_framework == "pytest":
                return self._generate_pytest_code(vector, import_prefix)
            elif test_framework == "unittest":
                return self._generate_unittest_code(vector, import_prefix)
            else:
                return self._generate_pytest_code(vector, import_prefix)
        elif language == Language.C:
            return self._generate_c_code(vector)
        elif language == Language.RUST:
            return self._generate_rust_code(vector)
        else:
            return self._generate_pytest_code(vector, import_prefix)  # Default
    
    def _generate_pytest_code(self, vector: TestVector, import_prefix: str) -> str:
        """Generate pytest test code"""
        module_import = f"{import_prefix}{vector.module_name}" if import_prefix else vector.module_name
        
        test_code = f'''"""
{vector.description}

Vector ID: {vector.vector_id}
Type: {vector.vector_type.value}
Priority: {vector.priority.value}
"""
import pytest
from {module_import} import *

'''
        
        # Add fixtures if needed
        if vector.setup_function:
            test_code += f'''
@pytest.fixture
def setup_{vector.vector_id}():
    """Setup fixture for {vector.vector_id}"""
    {vector.setup_function}()
    yield
    {vector.teardown_function or 'pass'}()
'''
        
        # Generate test function
        test_code += f'''
def test_{vector.vector_id}():
    """
    {vector.description}
    
    Coverage targets: {', '.join(vector.coverage_targets)}
    """
    # Setup
'''
        
        # Add preconditions
        for precondition in vector.preconditions:
            test_code += f"    # Precondition: {precondition}\n"
        
        if vector.setup_function:
            test_code += f"    setup_{vector.vector_id}()\n"
        
        # Add inputs
        if vector.inputs:
            test_code += "\n    # Inputs\n"
            for key, value in vector.inputs.items():
                test_code += f"    {key} = {repr(value)}\n"
        
        # Add test execution
        test_code += "\n    # Test execution\n"
        if vector.coverage_targets:
            # Try to call the first coverage target
            target = vector.coverage_targets[0]
            if '.' in target:
                # Class method
                class_name, method_name = target.split('.', 1)
                test_code += f"    # result = {class_name}().{method_name}({', '.join(vector.inputs.keys()) if vector.inputs else ''})\n"
            else:
                # Function
                test_code += f"    # result = {target}({', '.join(vector.inputs.keys()) if vector.inputs else ''})\n"
        
        # Add assertions
        test_code += "\n    # Assertions\n"
        if vector.expected_outputs:
            for key, value in vector.expected_outputs.items():
                test_code += f"    # assert {key} == {repr(value)}\n"
        
        if vector.expected_errors:
            test_code += "\n    # Error handling\n"
            for error in vector.expected_errors:
                test_code += f"    # with pytest.raises({error}):\n"
                test_code += f"    #     # Should raise {error}\n"
        
        # Add postconditions
        if vector.postconditions:
            test_code += "\n    # Postconditions\n"
            for postcondition in vector.postconditions:
                test_code += f"    # Postcondition: {postcondition}\n"
        
        test_code += "    pass  # TODO: Implement test\n"
        
        return test_code
    
    def _generate_unittest_code(self, vector: TestVector, import_prefix: str) -> str:
        """Generate unittest test code"""
        module_import = f"{import_prefix}{vector.module_name}" if import_prefix else vector.module_name
        
        test_code = f'''"""
{vector.description}

Vector ID: {vector.vector_id}
Type: {vector.vector_type.value}
Priority: {vector.priority.value}
"""
import unittest
from {module_import} import *


class Test{vector.vector_id.replace('_', '').title()}(unittest.TestCase):
    """Test case for {vector.vector_id}"""
    
    def setUp(self):
        """Set up test fixtures"""
'''
        
        for precondition in vector.preconditions:
            test_code += f"        # Precondition: {precondition}\n"
        
        if vector.setup_function:
            test_code += f"        {vector.setup_function}()\n"
        else:
            test_code += "        pass\n"
        
        test_code += f'''
    def test_{vector.vector_id}(self):
        """
        {vector.description}
        
        Coverage targets: {', '.join(vector.coverage_targets)}
        """
'''
        
        # Add inputs
        if vector.inputs:
            test_code += "        # Inputs\n"
            for key, value in vector.inputs.items():
                test_code += f"        {key} = {repr(value)}\n"
        
        # Add test execution
        test_code += "\n        # Test execution\n"
        test_code += "        # TODO: Implement test\n"
        
        # Add assertions
        if vector.expected_outputs:
            test_code += "\n        # Assertions\n"
            for key, value in vector.expected_outputs.items():
                test_code += f"        # self.assertEqual({key}, {repr(value)})\n"
        
        # Add teardown
        test_code += '''
    def tearDown(self):
        """Clean up after test"""
'''
        if vector.teardown_function:
            test_code += f"        {vector.teardown_function}()\n"
        else:
            test_code += "        pass\n"
        
        return test_code
    
    def _generate_c_code(self, vector: TestVector) -> str:
        """Generate C test code using Unity framework"""
        module_name = vector.module_name
        
        test_code = f'''/*
 * {vector.description}
 *
 * Vector ID: {vector.vector_id}
 * Type: {vector.vector_type.value}
 * Priority: {vector.priority.value}
 */

#include "unity.h"
#include "{module_name}.h"

void setUp(void) {{
    // Setup before each test
'''
        
        for precondition in vector.preconditions:
            test_code += f"    // Precondition: {precondition}\n"
        
        if vector.setup_function:
            test_code += f"    {vector.setup_function}();\n"
        
        test_code += "}\n\n"
        
        test_code += f"void tearDown(void) {{\n"
        if vector.teardown_function:
            test_code += f"    {vector.teardown_function}();\n"
        test_code += "}\n\n"
        
        # Generate test function
        test_func_name = f"test_{vector.vector_id.replace('-', '_')}"
        test_code += f"void {test_func_name}(void) {{\n"
        test_code += f"    /*\n"
        test_code += f"     * {vector.description}\n"
        test_code += f"     * Coverage targets: {', '.join(vector.coverage_targets)}\n"
        test_code += f"     */\n\n"
        
        # Add inputs
        if vector.inputs:
            test_code += "    // Inputs\n"
            for key, value in vector.inputs.items():
                if isinstance(value, str):
                    test_code += f"    const char* {key} = \"{value}\";\n"
                elif isinstance(value, int):
                    test_code += f"    int {key} = {value};\n"
                elif isinstance(value, float):
                    test_code += f"    float {key} = {value}f;\n"
                else:
                    test_code += f"    // {key} = {value}\n"
        
        # Add test execution
        test_code += "\n    // Test execution\n"
        if vector.coverage_targets:
            target = vector.coverage_targets[0]
            if '.' in target:
                # Struct method
                struct_name, method_name = target.split('.', 1)
                test_code += f"    // {struct_name} obj;\n"
                test_code += f"    // obj.{method_name}({', '.join(vector.inputs.keys()) if vector.inputs else ''});\n"
            else:
                # Function
                test_code += f"    // {target}({', '.join(vector.inputs.keys()) if vector.inputs else ''});\n"
        
        # Add assertions
        test_code += "\n    // Assertions\n"
        if vector.expected_outputs:
            for key, value in vector.expected_outputs.items():
                if isinstance(value, (int, float)):
                    test_code += f"    // TEST_ASSERT_EQUAL({value}, {key});\n"
                elif isinstance(value, str):
                    test_code += f"    // TEST_ASSERT_EQUAL_STRING(\"{value}\", {key});\n"
                else:
                    test_code += f"    // TEST_ASSERT_EQUAL({repr(value)}, {key});\n"
        
        if vector.expected_errors:
            test_code += "\n    // Error handling\n"
            for error in vector.expected_errors:
                test_code += f"    // Should handle {error}\n"
        
        # Add postconditions
        if vector.postconditions:
            test_code += "\n    // Postconditions\n"
            for postcondition in vector.postconditions:
                test_code += f"    // Postcondition: {postcondition}\n"
        
        test_code += "    // TODO: Implement test\n"
        test_code += "}\n"
        
        return test_code
    
    def _generate_rust_code(self, vector: TestVector) -> str:
        """Generate Rust test code"""
        module_name = vector.module_name
        
        test_code = f'''/*
 * {vector.description}
 *
 * Vector ID: {vector.vector_id}
 * Type: {vector.vector_type.value}
 * Priority: {vector.priority.value}
 */

#[cfg(test)]
mod tests {{
    use super::*;
    
'''
        
        # Generate test function
        test_func_name = f"test_{vector.vector_id.replace('-', '_')}"
        test_code += f"    #[test]\n"
        test_code += f"    fn {test_func_name}() {{\n"
        test_code += f"        /*\n"
        test_code += f"         * {vector.description}\n"
        test_code += f"         * Coverage targets: {', '.join(vector.coverage_targets)}\n"
        test_code += f"         */\n\n"
        
        # Add preconditions
        for precondition in vector.preconditions:
            test_code += f"        // Precondition: {precondition}\n"
        
        if vector.setup_function:
            test_code += f"        {vector.setup_function}();\n"
        
        # Add inputs
        if vector.inputs:
            test_code += "\n        // Inputs\n"
            for key, value in vector.inputs.items():
                if isinstance(value, str):
                    test_code += f"        let {key} = \"{value}\";\n"
                elif isinstance(value, int):
                    test_code += f"        let {key} = {value};\n"
                elif isinstance(value, float):
                    test_code += f"        let {key} = {value}f64;\n"
                elif isinstance(value, bool):
                    test_code += f"        let {key} = {str(value).lower()};\n"
                else:
                    test_code += f"        let {key} = {repr(value)};\n"
        
        # Add test execution
        test_code += "\n        // Test execution\n"
        if vector.coverage_targets:
            target = vector.coverage_targets[0]
            if '.' in target:
                # Method call
                struct_name, method_name = target.split('.', 1)
                test_code += f"        // let mut obj = {struct_name}::new();\n"
                test_code += f"        // let result = obj.{method_name}({', '.join(vector.inputs.keys()) if vector.inputs else ''});\n"
            else:
                # Function call
                test_code += f"        // let result = {target}({', '.join(vector.inputs.keys()) if vector.inputs else ''});\n"
        
        # Add assertions
        test_code += "\n        // Assertions\n"
        if vector.expected_outputs:
            for key, value in vector.expected_outputs.items():
                if isinstance(value, str):
                    test_code += f"        // assert_eq!({key}, \"{value}\");\n"
                elif isinstance(value, (int, float)):
                    test_code += f"        // assert_eq!({key}, {value});\n"
                elif isinstance(value, bool):
                    test_code += f"        // assert_eq!({key}, {str(value).lower()});\n"
                else:
                    test_code += f"        // assert_eq!({key}, {repr(value)});\n"
        
        if vector.expected_errors:
            test_code += "\n        // Error handling\n"
            for error in vector.expected_errors:
                test_code += f"        // Should handle {error}\n"
                test_code += f"        // assert!(matches!(result, Err({error})));\n"
        
        # Add postconditions
        if vector.postconditions:
            test_code += "\n        // Postconditions\n"
            for postcondition in vector.postconditions:
                test_code += f"        // Postcondition: {postcondition}\n"
        
        if vector.teardown_function:
            test_code += f"\n        {vector.teardown_function}();\n"
        
        test_code += "        // TODO: Implement test\n"
        test_code += "    }\n"
        test_code += "}\n"
        
        return test_code
    
    def save_generated_tests(self, output_dir: Path):
        """Save generated test code to files"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Group by module
        by_module = {}
        for vector in self.generated_vectors:
            if vector.module_name not in by_module:
                by_module[vector.module_name] = []
            by_module[vector.module_name].append(vector)
        
        for module_name, vectors in by_module.items():
            # Detect language from first vector's module
            module_path = self.coverage_analyzer._find_module_file(module_name)
            if module_path:
                language = self.language_parser.detect_language(module_path)
            else:
                language = Language.PYTHON  # Default
            
            # Generate test code for all vectors
            test_code = ""
            if language == Language.PYTHON:
                test_code = f'''"""
Auto-generated tests for {module_name}
Generated by Unified Test Harness
"""
'''
            elif language == Language.C:
                test_code = f'''/*
 * Auto-generated tests for {module_name}
 * Generated by Unified Test Harness
 */

#include "unity.h"
#include "{module_name}.h"

'''
            elif language == Language.RUST:
                test_code = f'''/*
 * Auto-generated tests for {module_name}
 * Generated by Unified Test Harness
 */

#[cfg(test)]
mod tests {{
    use super::*;

'''
            
            for vector in vectors:
                vector_code = self.generate_test_code(vector)
                test_code += vector_code
                test_code += "\n\n"
            
            if language == Language.RUST:
                test_code += "}\n"
            
            # Determine file extension
            if language == Language.PYTHON:
                test_file = output_dir / f"test_{module_name}_generated.py"
            elif language == Language.C:
                test_file = output_dir / f"test_{module_name}_generated.c"
            elif language == Language.RUST:
                test_file = output_dir / f"test_{module_name}_generated.rs"
            else:
                test_file = output_dir / f"test_{module_name}_generated.py"
            
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            print(f"[+] Saved {len(vectors)} tests to {test_file}")
