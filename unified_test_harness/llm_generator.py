"""
LLM-Powered Test Generator

Uses LLM agents to generate test vectors based on coverage gaps and code analysis.
Framework-agnostic implementation.
Supports Python, C, and Rust test generation.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import ast

from .test_vector import TestVector, TestVectorType, TestPriority
from .language_parser import LanguageParser, Language
from .code_analyzer import CodeAnalyzer, FunctionInfo

# Configure logging
logger = logging.getLogger(__name__)


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
        self.code_analyzer = CodeAnalyzer()
        
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
                logger.warning("OpenAI library not installed. Install with: pip install openai")
                logger.warning("LLM features will be disabled")
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
                logger.warning("Anthropic library not installed. Install with: pip install anthropic")
                logger.warning("LLM features will be disabled")
        else:
            logger.error(f"LLM provider '{provider}' not supported")
    
    def generate_test_vector_prompt(self, module_name: str, uncovered_functions: List[str], 
                                   module_code: str, file_path: str) -> str:
        """Generate enhanced prompt for LLM to create test vectors with detailed context"""
        import_prefix = self.config.framework.import_prefix
        
        safety_directives = []
        if not getattr(self.config, "allow_network", False):
            safety_directives.append("Do NOT perform real network calls; mock sockets/HTTP clients.")
        if not getattr(self.config, "allow_filesystem", False):
            safety_directives.append("Do NOT perform real filesystem writes; use tmp paths or mocks.")
        safety_directives.append("Use deterministic seeds where randomness is needed.")
        safety_directives.append("Avoid sleeps/time-dependent assertions; keep tests fast and deterministic.")
        safety_text = "\n".join([f"- {d}" for d in safety_directives])

        integration_targets = getattr(self.config, "integration_targets", [])
        dependency_stubs = getattr(self.config, "dependency_stubs", {})
        focus_paths = getattr(self.config, "focus_paths", [])

        # Analyze functions to get detailed information
        module_path = self.coverage_analyzer._find_module_file(module_name)
        function_details = []
        similar_tests = []
        
        module_info = None
        if module_path:
            module_info = self.code_analyzer.analyze_module(module_path)
            
            # Get detailed function information
            for func_name in uncovered_functions:
                # Check if it's a method (Class.method)
                if '.' in func_name:
                    class_name, method_name = func_name.split('.', 1)
                    func_info = self.code_analyzer.analyze_function(module_path, method_name, class_name)
                else:
                    func_info = self.code_analyzer.analyze_function(module_path, func_name)
                
                if func_info:
                    function_details.append(func_info)
            
            # Find similar test patterns from vector database
            if self.config.use_vector_db and self.code_embedder:
                for func_info in function_details[:3]:  # Limit to first 3 for examples
                    similar = self.code_embedder.find_similar_tests(func_info.code, n_results=2)
                    if similar and 'documents' in similar:
                        similar_tests.extend(similar['documents'][:2])
        
        # Build function details section
        func_details_text = ""
        for func_info in function_details:
            func_details_text += f"\nFunction: {func_info.name}\n"
            func_details_text += f"  Signature: {func_info.signature}\n"
            if func_info.parameters:
                func_details_text += f"  Parameters:\n"
                for param in func_info.parameters:
                    param_str = f"    - {param.get('name', 'unknown')}"
                    if 'type' in param:
                        param_str += f" ({param['type']})"
                    if 'default' in param:
                        param_str += f" = {param['default']}"
                    func_details_text += param_str + "\n"
            if func_info.return_type:
                func_details_text += f"  Returns: {func_info.return_type}\n"
            if func_info.docstring:
                func_details_text += f"  Docstring: {func_info.docstring[:200]}...\n"
            if func_info.raises:
                func_details_text += f"  Raises: {', '.join(func_info.raises)}\n"
            if func_info.dependencies:
                func_details_text += f"  Uses: {', '.join(func_info.dependencies[:5])}\n"
            if func_info.io_operations:
                func_details_text += f"  IO: {', '.join(func_info.io_operations[:5])}\n"
            if func_info.network_calls:
                func_details_text += f"  Network: {', '.join(func_info.network_calls[:5])}\n"
            if func_info.env_usage:
                func_details_text += f"  Env: {', '.join(func_info.env_usage[:5])}\n"
            if func_info.global_state:
                func_details_text += f"  Globals: {', '.join(func_info.global_state[:5])}\n"
        
        # Build similar tests examples
        examples_text = ""
        if similar_tests:
            examples_text = "\n\n## Example Test Patterns from Similar Code:\n\n"
            for i, test_code in enumerate(similar_tests[:3], 1):
                examples_text += f"Example {i}:\n```\n{test_code[:300]}...\n```\n\n"
        
        prompt = f"""You are an expert test engineer. Generate comprehensive, production-ready test vectors for the following code.

## Module Information
- Module Name: {module_name}
- File Path: {file_path}
- Import Prefix: {import_prefix}
- Test Framework: {self.config.framework.test_framework}
- Language: {module_info.language.value if module_path else 'python'}

## Functions to Test
{', '.join(uncovered_functions)}

## Detailed Function Information
{func_details_text}

## Module Code
```{module_info.language.value if module_path else 'python'}
{module_code[:2000]}
```
{examples_text}
## Requirements

Generate test vectors that:

1. **Coverage**: Create test vectors for each uncovered function listed above
2. **Completeness**: Include:
   - Unit tests with normal inputs
   - Edge cases (boundary values, empty inputs, None/null, etc.)
   - Error handling tests for all exceptions listed in function details
   - Integration tests if function has dependencies
3. **Accuracy**: 
   - Use actual function signatures and parameter types from the function details
   - Match return types exactly
   - Test all exception paths identified in "Raises"
   - Use realistic test data based on parameter types
4. **Quality**:
   - Set priority: "critical" for core business logic, "high" for important utilities, "medium" for helpers, "low" for edge cases
   - Include clear descriptions explaining what each test validates
   - Add meaningful tags (e.g., "unit", "integration", "edge_case", "error_handling")
5. **Test Data**:
   - For numeric types: include 0, negative, positive, boundary values
   - For strings: include empty string, normal string, special characters
   - For collections: include empty, single item, multiple items
   - For optional/nullable: include None/null and valid values
   - Use realistic values based on function purpose and docstring
6. **Safety & Determinism**:
   - {safety_text}
   - Use dependency stubs when present: {json.dumps(dependency_stubs)}
   - Prefer local/temp data; avoid real endpoints, secrets, or filesystem mutations outside tmp dirs.
7. **Integration Targets**:
   - Treat modules in {integration_targets} or paths under {focus_paths} as integration candidates; include fakes for network/FS/IPC boundaries.

## Output Format

Return ONLY valid JSON (no markdown, no code blocks):
{{
    "vectors": [
        {{
            "vector_id": "module_function_test_type",
            "name": "Descriptive test name",
            "description": "Clear description of what this test validates and why",
            "module_name": "{module_name}",
            "vector_type": "unit|integration|edge_case|error_handling",
            "priority": "critical|high|medium|low",
            "inputs": {{"param1": "actual_value", "param2": 42}},
            "expected_outputs": {{"result": "expected_value"}},
            "expected_errors": ["SpecificErrorType"],
            "coverage_targets": ["function_name"],
            "coverage_minimum": 0.8,
            "preconditions": ["state_required"],
            "postconditions": ["state_after_test"],
            "tags": ["unit", "edge_case"]
        }}
    ]
}}

## Important Notes
- Use actual parameter names from function signatures
- Match types exactly (e.g., if parameter is int, use integer, not string)
- Include all parameters in inputs (don't skip optional ones unless testing default behavior)
- For error tests, use exact exception types from function details
- Make test names descriptive and specific
- Ensure expected_outputs match return_type from function details
"""
        return prompt
    
    def _call_llm(self, prompt: str, refinement: bool = False) -> str:
        """Call LLM API with enhanced system message"""
        if not self.llm_client:
            return ""
        
        provider = self.config.llm_provider.lower()
        
        # Enhanced system message
        system_message = """You are an expert test engineer with deep knowledge of software testing best practices.

Your task is to generate high-quality, production-ready test vectors that:
1. Are accurate and complete - use actual function signatures, types, and behavior
2. Cover all important scenarios - normal cases, edge cases, error cases
3. Use realistic test data appropriate for the function's purpose
4. Follow testing best practices and patterns from the codebase
5. Avoid external side-effects (network/filesystem) unless explicitly allowed; prefer mocks/fakes.
6. Use deterministic seeds; avoid sleeps/time-sensitive behavior.
7. Output valid JSON that can be directly used without manual correction

When generating test vectors:
- Analyze function signatures carefully and use correct parameter names and types
- Read docstrings to understand function purpose and expected behavior
- Consider all exception types that might be raised
- Use appropriate test data (e.g., for int use numbers, for str use strings)
- Make test names descriptive and specific
- Ensure expected outputs match return types exactly
- Include edge cases like empty inputs, None/null, boundary values
- Test error conditions for all exceptions listed

Output ONLY valid JSON. Do not include markdown code blocks or explanations outside the JSON."""
        
        if refinement:
            system_message += "\n\nYou are now refining and validating generated test vectors. Check for:\n"
            system_message += "- Type mismatches between inputs and function parameters\n"
            system_message += "- Missing required parameters\n"
            system_message += "- Incorrect expected outputs (wrong type or value)\n"
            system_message += "- Missing edge cases or error handling\n"
            system_message += "- Improve test descriptions and names for clarity"
        
        provider = self.config.llm_provider.lower()
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                if provider == "openai":
                    response = self.llm_client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3 if refinement else 0.7,  # Lower temperature for refinement
                        max_tokens=4000
                    )
                    return response.choices[0].message.content
                elif provider == "anthropic":
                    response = self.llm_client.messages.create(
                        model=self.config.llm_model,
                        max_tokens=4000,
                        messages=[
                            {"role": "user", "content": f"{system_message}\n\n{prompt}"}
                        ],
                        temperature=0.3 if refinement else 0.7
                    )
                    return response.content[0].text
            except Exception as e:
                if attempt < max_retries - 1:
                    # Exponential backoff: wait 2^attempt seconds
                    wait_time = 2 ** attempt
                    logger.warning(f"LLM API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM API error after {max_retries} attempts: {e}", exc_info=True)
                    return ""
        
        return ""
    
    def parse_llm_response(self, response: str) -> List[TestVector]:
        """Parse LLM response and create test vectors with validation"""
        try:
            # Extract JSON from response - try multiple methods
            json_str = None
            
            # Method 1: Look for JSON code block
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Method 2: Find first { to last }
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_str = response[json_start:json_end]
            
            if not json_str:
                logger.warning("Could not extract JSON from LLM response")
                return []
            
            data = json.loads(json_str)
            
            vectors = []
            for vec_data in data.get('vectors', []):
                try:
                    # Validate vector data
                    if not self._validate_vector_data(vec_data):
                        logger.warning(f"Skipping invalid vector: {vec_data.get('vector_id', 'unknown')}")
                        continue
                    if self._contains_external_reference(vec_data):
                        logger.warning(f"Skipping vector with external references: {vec_data.get('vector_id', 'unknown')}")
                        continue
                    
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
                    logger.warning(f"Error parsing vector: {e}")
                    continue
            
            return vectors
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}", exc_info=True)
            return []
    
    def _validate_vector_data(self, vec_data: Dict[str, Any]) -> bool:
        """Validate vector data for correctness"""
        required_fields = ['vector_id', 'name', 'description', 'module_name', 
                          'vector_type', 'priority', 'coverage_targets']
        
        for field in required_fields:
            if field not in vec_data:
                return False
        
        # Validate vector_type
        try:
            TestVectorType(vec_data['vector_type'])
        except ValueError:
            return False
        
        # Validate priority
        try:
            TestPriority(vec_data['priority'])
        except ValueError:
            return False
        
        # Ensure coverage_targets is a list
        if not isinstance(vec_data.get('coverage_targets', []), list):
            return False
        
        return True

    def _contains_external_reference(self, vec_data: Dict[str, Any]) -> bool:
        """Guardrail to avoid network/secret usage in generated vectors"""
        check_strings = []
        for val in vec_data.get('inputs', {}).values():
            if isinstance(val, str):
                check_strings.append(val)
        for val in vec_data.get('expected_outputs', {}).values():
            if isinstance(val, str):
                check_strings.append(val)
        banned_prefixes = ("http://", "https://", "ssh://")
        for text in check_strings:
            if text.startswith(banned_prefixes):
                return True
        return False
    
    def refine_vectors(self, vectors: List[TestVector]) -> List[TestVector]:
        """Refine generated vectors using LLM validation"""
        if not self.llm_client or not vectors:
            return vectors
        
        # Get function details for validation
        if not vectors:
            return vectors
        
        module_name = vectors[0].module_name
        module_path = self.coverage_analyzer._find_module_file(module_name)
        
        if not module_path:
            return vectors
        
        # Build refinement prompt
        vectors_json = json.dumps([v.to_dict() for v in vectors], indent=2)
        
        refinement_prompt = f"""Review and refine the following test vectors. Fix any issues with:
1. Type mismatches (e.g., passing string to int parameter)
2. Missing required parameters
3. Incorrect expected outputs (wrong type or unrealistic values)
4. Missing edge cases
5. Unclear descriptions

Module: {module_name}
Module Path: {module_path}

Generated Vectors:
{vectors_json}

Return the refined vectors in the same JSON format, fixing all issues found."""
        
        try:
            refined_response = self._call_llm(refinement_prompt, refinement=True)
            if refined_response:
                refined_vectors = self.parse_llm_response(refined_response)
                if refined_vectors:
                    logger.info(f"Refined {len(refined_vectors)} vectors")
                    return refined_vectors
        except Exception as e:
            logger.error(f"Error refining vectors: {e}", exc_info=True)
        
        return vectors
    
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
            logger.error(f"Error reading module {module_name}: {e}", exc_info=True)
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
                # Refine vectors to improve quality
                if vectors:
                    vectors = self.refine_vectors(vectors)
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
            if module_name in getattr(self.config, "integration_targets", []):
                vector_type = TestVectorType.INTEGRATION
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
        """Generate complete, executable pytest test code"""
        module_import = f"{import_prefix}{vector.module_name}" if import_prefix else vector.module_name
        
        # Get function details for better code generation
        module_path = self.coverage_analyzer._find_module_file(vector.module_name)
        func_info = None
        if module_path and vector.coverage_targets:
            target = vector.coverage_targets[0]
            if '.' in target:
                class_name, method_name = target.split('.', 1)
                func_info = self.code_analyzer.analyze_function(module_path, method_name, class_name)
            else:
                func_info = self.code_analyzer.analyze_function(module_path, target)
        
        test_code = f'''"""
{vector.description}

Vector ID: {vector.vector_id}
Type: {vector.vector_type.value}
Priority: {vector.priority.value}
Coverage targets: {', '.join(vector.coverage_targets)}
"""
import pytest
import random
random.seed({getattr(self.config, "random_seed", 1337)})
'''
        if not getattr(self.config, "allow_network", False):
            test_code += '''
@pytest.fixture(autouse=True)
def _disable_network(monkeypatch):
    import socket
    def _blocked(*args, **kwargs):
        raise RuntimeError("network disabled in generated tests")
    monkeypatch.setattr(socket, "create_connection", _blocked, raising=False)
'''
        if not getattr(self.config, "allow_filesystem", False):
            test_code += '''
@pytest.fixture(autouse=True)
def _guard_filesystem(monkeypatch, tmp_path):
    import builtins, os
    real_open = builtins.open
    def _safe_open(file, mode='r', *args, **kwargs):
        if any(flag in mode for flag in ('w', 'a', 'x')) and not str(file).startswith(str(tmp_path)):
            raise RuntimeError("filesystem writes disabled in generated tests")
        return real_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", _safe_open)
    monkeypatch.setenv("TMPDIR", str(tmp_path))
'''
        
        # Import specific items if we have function info
        if func_info and func_info.parent_class:
            test_code += f"from {module_import} import {func_info.parent_class}\n"
        elif func_info:
            test_code += f"from {module_import} import {func_info.name}\n"
        else:
            test_code += f"from {module_import} import *\n"
        
        test_code += "\n"
        
        # Add fixtures if needed
        if vector.setup_function:
            test_code += f'''
@pytest.fixture
def setup_{vector.vector_id.replace('-', '_')}():
    """Setup fixture for {vector.vector_id}"""
    {vector.setup_function}()
    yield
    {vector.teardown_function or 'pass'}()
'''
        
        # Generate test function
        test_func_name = f"test_{vector.vector_id.replace('-', '_')}"
        test_code += f'''
def {test_func_name}():
    """
    {vector.description}
    
    Coverage targets: {', '.join(vector.coverage_targets)}
    '''
        
        if vector.preconditions:
            test_code += "\n    Preconditions:\n"
            for precondition in vector.preconditions:
                test_code += f"    - {precondition}\n"
        
        if vector.postconditions:
            test_code += "\n    Postconditions:\n"
            for postcondition in vector.postconditions:
                test_code += f"    - {postcondition}\n"
        
        test_code += '    """\n'
        
        # Add preconditions setup
        if vector.preconditions:
            test_code += "    # Setup preconditions\n"
            for precondition in vector.preconditions:
                test_code += f"    # Ensure: {precondition}\n"
        
        if vector.setup_function:
            test_code += f"    setup_{vector.vector_id.replace('-', '_')}()\n"
        
        # Add inputs with proper types
        if vector.inputs:
            test_code += "\n    # Prepare test inputs\n"
            for key, value in vector.inputs.items():
                # Use proper Python representation
                if isinstance(value, str):
                    test_code += f"    {key} = {repr(value)}\n"
                elif isinstance(value, (int, float, bool)):
                    test_code += f"    {key} = {value}\n"
                elif value is None:
                    test_code += f"    {key} = None\n"
                else:
                    test_code += f"    {key} = {repr(value)}\n"
        
        # Generate actual test execution
        test_code += "\n    # Execute test\n"
        if vector.coverage_targets:
            target = vector.coverage_targets[0]
            
            # Build function call
            if '.' in target:
                # Class method
                class_name, method_name = target.split('.', 1)
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"    instance = {class_name}()\n"
                    test_code += f"    result = instance.{method_name}({args})\n"
                else:
                    test_code += f"    instance = {class_name}()\n"
                    test_code += f"    result = instance.{method_name}()\n"
            else:
                # Function
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"    result = {target}({args})\n"
                else:
                    test_code += f"    result = {target}()\n"
        else:
            test_code += "    # No coverage target specified\n"
            test_code += "    result = None\n"
        
        # Generate assertions
        test_code += "\n    # Verify results\n"
        if vector.expected_errors:
            # Error handling test
            for error in vector.expected_errors:
                error_type = error.split('.')[-1]  # Get just the class name
                test_code += f"    with pytest.raises({error_type}):\n"
                if vector.coverage_targets:
                    target = vector.coverage_targets[0]
                    if '.' in target:
                        class_name, method_name = target.split('.', 1)
                        if vector.inputs:
                            args = ', '.join(vector.inputs.keys())
                            test_code += f"        instance = {class_name}()\n"
                            test_code += f"        instance.{method_name}({args})\n"
                        else:
                            test_code += f"        instance = {class_name}()\n"
                            test_code += f"        instance.{method_name}()\n"
                    else:
                        if vector.inputs:
                            args = ', '.join(vector.inputs.keys())
                            test_code += f"        {target}({args})\n"
                        else:
                            test_code += f"        {target}()\n"
        elif vector.expected_outputs:
            # Normal assertions
            for key, value in vector.expected_outputs.items():
                if key == 'result':
                    # Direct result assertion
                    if isinstance(value, str):
                        test_code += f"    assert result == {repr(value)}\n"
                    elif isinstance(value, (int, float, bool)):
                        test_code += f"    assert result == {value}\n"
                    elif value is None:
                        test_code += f"    assert result is None\n"
                    else:
                        test_code += f"    assert result == {repr(value)}\n"
                else:
                    # Assertion on a variable
                    if isinstance(value, str):
                        test_code += f"    assert {key} == {repr(value)}\n"
                    elif isinstance(value, (int, float, bool)):
                        test_code += f"    assert {key} == {value}\n"
                    else:
                        test_code += f"    assert {key} == {repr(value)}\n"
        else:
            # No expected outputs, just verify no exception
            test_code += "    assert result is not None  # Function executed successfully\n"
        
        # Add postcondition checks
        if vector.postconditions:
            test_code += "\n    # Verify postconditions\n"
            for postcondition in vector.postconditions:
                test_code += f"    # Postcondition: {postcondition}\n"
        
        test_code += "\n"
        
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
        test_code += "\n        # Execute test\n"
        if vector.coverage_targets:
            target = vector.coverage_targets[0]
            if '.' in target:
                class_name, method_name = target.split('.', 1)
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"        instance = {class_name}()\n"
                    test_code += f"        result = instance.{method_name}({args})\n"
                else:
                    test_code += f"        instance = {class_name}()\n"
                    test_code += f"        result = instance.{method_name}()\n"
            else:
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"        result = {target}({args})\n"
                else:
                    test_code += f"        result = {target}()\n"
        
        # Add assertions
        if vector.expected_outputs:
            test_code += "\n        # Verify results\n"
            for key, value in vector.expected_outputs.items():
                if key == 'result':
                    if isinstance(value, str):
                        test_code += f"        self.assertEqual(result, {repr(value)})\n"
                    elif isinstance(value, (int, float, bool)):
                        test_code += f"        self.assertEqual(result, {value})\n"
                    else:
                        test_code += f"        self.assertEqual(result, {repr(value)})\n"
                else:
                    if isinstance(value, str):
                        test_code += f"        self.assertEqual({key}, {repr(value)})\n"
                    else:
                        test_code += f"        self.assertEqual({key}, {repr(value)})\n"
        
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
        """Generate C test code using Unity or CMocka framework"""
        module_name = vector.module_name
        
        # Check framework preference from config
        test_framework = self.config.framework.test_framework.lower()
        use_cmocka = test_framework == "cmocka" or "cmocka" in test_framework
        
        if use_cmocka:
            return self._generate_cmocka_code(vector, module_name)
        else:
            return self._generate_unity_code(vector, module_name)
    
    def _generate_unity_code(self, vector: TestVector, module_name: str) -> str:
        """Generate C test code using Unity framework"""
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
        test_code += "\n    // Execute test\n"
        if vector.coverage_targets:
            target = vector.coverage_targets[0]
            if '.' in target:
                # Struct method
                struct_name, method_name = target.split('.', 1)
                test_code += f"    {struct_name} obj;\n"
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"    // Initialize struct with: {args}\n"
                    test_code += f"    {method_name}(&obj, {args});\n"
                else:
                    test_code += f"    {method_name}(&obj);\n"
                test_code += "    // result = obj;\n"
            else:
                # Function
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"    int result = {target}({args});\n"
                else:
                    test_code += f"    int result = {target}();\n"
        
        # Add assertions
        test_code += "\n    // Verify results\n"
        if vector.expected_outputs:
            for key, value in vector.expected_outputs.items():
                if isinstance(value, (int, float)):
                    test_code += f"    TEST_ASSERT_EQUAL({value}, result);\n"
                elif isinstance(value, str):
                    test_code += f"    TEST_ASSERT_EQUAL_STRING(\"{value}\", result);\n"
                else:
                    test_code += f"    // TEST_ASSERT_EQUAL({repr(value)}, result);\n"
        
        if vector.expected_errors:
            test_code += "\n    // Error handling\n"
            for error in vector.expected_errors:
                test_code += f"    // Expected error: {error}\n"
        
        # Add postconditions
        if vector.postconditions:
            test_code += "\n    // Verify postconditions\n"
            for postcondition in vector.postconditions:
                test_code += f"    // Postcondition: {postcondition}\n"
        test_code += "}\n"
        
        return test_code
    
    def _generate_cmocka_code(self, vector: TestVector, module_name: str) -> str:
        """Generate C test code using CMocka framework"""
        test_code = f'''/*
 * {vector.description}
 *
 * Vector ID: {vector.vector_id}
 * Type: {vector.vector_type.value}
 * Priority: {vector.priority.value}
 */

#include <stdarg.h>
#include <stddef.h>
#include <setjmp.h>
#include <cmocka.h>
#include "{module_name}.h"

static int setup(void **state) {{
    // Setup before each test
'''
        
        for precondition in vector.preconditions:
            test_code += f"    // Precondition: {precondition}\n"
        
        if vector.setup_function:
            test_code += f"    {vector.setup_function}();\n"
        
        test_code += "    return 0;\n}\n\n"
        
        test_code += f"static int teardown(void **state) {{\n"
        if vector.teardown_function:
            test_code += f"    {vector.teardown_function}();\n"
        test_code += "    return 0;\n}\n\n"
        
        # Generate test function
        test_func_name = f"test_{vector.vector_id.replace('-', '_')}"
        test_code += f"static void {test_func_name}(void **state) {{\n"
        test_code += f"    (void)state;  // Unused parameter\n"
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
        test_code += "\n    // Execute test\n"
        if vector.coverage_targets:
            target = vector.coverage_targets[0]
            if '.' in target:
                # Struct method
                struct_name, method_name = target.split('.', 1)
                test_code += f"    {struct_name} obj;\n"
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"    // Initialize struct with: {args}\n"
                    test_code += f"    {method_name}(&obj, {args});\n"
                else:
                    test_code += f"    {method_name}(&obj);\n"
            else:
                # Function call
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"    // result = {target}({args});\n"
                else:
                    test_code += f"    // result = {target}();\n"
        
        # Add assertions
        if vector.expected_outputs:
            test_code += "\n    // Assertions\n"
            for key, value in vector.expected_outputs.items():
                if isinstance(value, (int, float)):
                    test_code += f"    assert_int_equal({key}, {value});\n"
                elif isinstance(value, str):
                    test_code += f"    assert_string_equal({key}, \"{value}\");\n"
                else:
                    test_code += f"    // assert({key} == {repr(value)});\n"
        
        test_code += "}\n\n"
        
        # Main function
        test_code += f"int main(void) {{\n"
        test_code += f"    const struct CMUnitTest tests[] = {{\n"
        test_code += f"        cmocka_unit_test_setup_teardown({test_func_name}, setup, teardown),\n"
        test_code += f"    }};\n"
        test_code += f"    return cmocka_run_group_tests(tests, NULL, NULL);\n"
        test_code += f"}}\n"
        
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
        test_code += "\n        // Execute test\n"
        if vector.coverage_targets:
            target = vector.coverage_targets[0]
            if '.' in target:
                # Method call
                struct_name, method_name = target.split('.', 1)
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"        let mut obj = {struct_name}::new();\n"
                    test_code += f"        let result = obj.{method_name}({args});\n"
                else:
                    test_code += f"        let mut obj = {struct_name}::new();\n"
                    test_code += f"        let result = obj.{method_name}();\n"
            else:
                # Function call
                if vector.inputs:
                    args = ', '.join(vector.inputs.keys())
                    test_code += f"        let result = {target}({args});\n"
                else:
                    test_code += f"        let result = {target}();\n"
        
        # Add assertions
        test_code += "\n        // Verify results\n"
        if vector.expected_errors:
            # Error handling test
            for error in vector.expected_errors:
                test_code += f"        assert!(matches!(result, Err({error})));\n"
        elif vector.expected_outputs:
            for key, value in vector.expected_outputs.items():
                if key == 'result':
                    if isinstance(value, str):
                        test_code += f"        assert_eq!(result, \"{value}\");\n"
                    elif isinstance(value, (int, float)):
                        test_code += f"        assert_eq!(result, {value});\n"
                    elif isinstance(value, bool):
                        test_code += f"        assert_eq!(result, {str(value).lower()});\n"
                    else:
                        test_code += f"        assert_eq!(result, {repr(value)});\n"
                else:
                    if isinstance(value, str):
                        test_code += f"        assert_eq!({key}, \"{value}\");\n"
                    elif isinstance(value, (int, float)):
                        test_code += f"        assert_eq!({key}, {value});\n"
                    else:
                        test_code += f"        assert_eq!({key}, {repr(value)});\n"
        else:
            test_code += "        assert!(result.is_ok());  // Function executed successfully\n"
        
        # Add postconditions
        if vector.postconditions:
            test_code += "\n        // Verify postconditions\n"
            for postcondition in vector.postconditions:
                test_code += f"        // Postcondition: {postcondition}\n"
        
        if vector.teardown_function:
            test_code += f"\n        {vector.teardown_function}();\n"
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
                # Validate syntax early to reduce post-editing
                try:
                    import ast as _ast
                    _ast.parse(test_code)
                except Exception as e:  # pragma: no cover - defensive
                    test_code = f"# Syntax check failed: {e}\n" + test_code
            elif language == Language.C:
                test_file = output_dir / f"test_{module_name}_generated.c"
            elif language == Language.RUST:
                test_file = output_dir / f"test_{module_name}_generated.rs"
            else:
                test_file = output_dir / f"test_{module_name}_generated.py"
            
            with open(test_file, 'w') as f:
                f.write(test_code)
            
            logger.info(f"Saved {len(vectors)} tests to {test_file}")
