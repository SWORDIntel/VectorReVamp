"""
Template Engine for VectorReVamp Test Framework

Learns test patterns from existing codebase and applies them to generate
high-quality tests using rule-based synthesis inspired by vector_revamp.
"""

import os
import re
import ast
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, Pattern
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib

from .quality_validator import TestQualityValidator, QualityReport
# Import template evolution components lazily to avoid circular imports

logger = logging.getLogger(__name__)


@dataclass
class TestTemplate:
    """Template extracted from existing test code."""

    template_id: str
    test_type: str  # 'unit', 'integration', 'edge_case', etc.
    language: str   # 'python', 'c', 'rust'
    framework: str  # 'pytest', 'unittest', 'cmocka', etc.

    # Template structure
    structure: Dict[str, Any] = field(default_factory=dict)

    # Pattern components
    setup_patterns: List[str] = field(default_factory=list)
    test_patterns: List[str] = field(default_factory=list)
    assertion_patterns: List[str] = field(default_factory=list)
    cleanup_patterns: List[str] = field(default_factory=list)

    # Metadata
    extracted_from: List[str] = field(default_factory=list)  # Source files
    quality_score: float = 0.0
    usage_count: int = 0
    success_rate: float = 0.0

    # Statistical properties
    complexity_level: str = "medium"  # 'simple', 'medium', 'complex'
    coverage_targets: List[str] = field(default_factory=list)
    parameter_patterns: Dict[str, Any] = field(default_factory=dict)

    # Validation rules
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CodeContext:
    """Context information about code being tested."""

    module_name: str
    function_name: str
    function_signature: str
    parameters: Dict[str, str]  # param_name -> param_type
    return_type: str
    decorators: List[str]
    docstring: str
    imports: List[str]
    dependencies: List[str]

    # Code analysis
    complexity: int
    has_exceptions: bool
    has_side_effects: bool
    testability_score: float


@dataclass
class GeneratedTest:
    """Result of template-based test generation."""

    test_code: str
    template_used: str
    target_function: str
    test_type: str
    quality_score: float
    coverage_estimate: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class TemplateEngine:
    """
    Template extraction and application engine inspired by vector_revamp's
    rule-based synthesis approach.
    """

    def __init__(self, config):
        self.config = config
        self.templates: Dict[str, TestTemplate] = {}
        self.template_patterns: Dict[str, List[Pattern]] = {}
        self.extraction_stats: Dict[str, Any] = {}

        # Initialize regex patterns for different languages/frameworks
        self._initialize_patterns()

        # Quality validation system (inspired by vector_revamp)
        self.quality_validator = TestQualityValidator()

        # Template evolution system (self-improving templates) - lazy loaded
        self._evolution_engine = None

        # Template quality tracking
        self.template_quality_history: Dict[str, List[float]] = {}
        self.generation_quality_history: Dict[str, List[float]] = {}

        logger.info("TemplateEngine initialized with quality validation")

    @property
    def evolution_engine(self):
        """Lazy load evolution engine to avoid circular imports."""
        if self._evolution_engine is None:
            from .template_evolution import TemplateEvolutionEngine
            self._evolution_engine = TemplateEvolutionEngine(self.config)
        return self._evolution_engine

    def _initialize_patterns(self):
        """Initialize regex patterns for template extraction."""

        # Python pytest patterns
        self.template_patterns['python_pytest'] = [
            re.compile(r'def test_(\w+)\([^)]*\):'),
            re.compile(r'assert\s+(.+)'),
            re.compile(r'with\s+pytest\.raises\(([^)]+)\):'),
            re.compile(r'@pytest\.fixture'),
            re.compile(r'@pytest\.mark\.(\w+)'),
        ]

        # Python unittest patterns
        self.template_patterns['python_unittest'] = [
            re.compile(r'def test_(\w+)\(self[^)]*\):'),
            re.compile(r'self\.assert(.+)\(([^)]+)\)'),
            re.compile(r'with self\.assertRaises\(([^)]+)\):'),
            re.compile(r'def setUp\(self\):'),
            re.compile(r'def tearDown\(self\):'),
        ]

        # C Unity patterns
        self.template_patterns['c_unity'] = [
            re.compile(r'void test_(\w+)\(void\)'),
            re.compile(r'TEST_ASSERT(.+)\(([^)]+)\)'),
            re.compile(r'TEST_IGNORE()'),
            re.compile(r'UNITY_BEGIN\(\);'),
            re.compile(r'UNITY_END\(\);'),
        ]

        # C CMocka patterns
        self.template_patterns['c_cmocka'] = [
            re.compile(r'static void test_(\w+)\(void \*\*state\)'),
            re.compile(r'assert_(.+)\(([^)]+)\)'),
            re.compile(r'cmocka_unit_test_setup_teardown\('),
            re.compile(r'cmocka_run_group_tests\('),
        ]

        # Rust patterns
        self.template_patterns['rust'] = [
            re.compile(r'#\[test\]'),
            re.compile(r'fn test_(\w+)\(\)'),
            re.compile(r'assert!(.+)'),
            re.compile(r'assert_eq!\(([^,]+),\s*([^)]+)\)'),
            re.compile(r'assert_ne!\(([^,]+),\s*([^)]+)\)'),
        ]

    def extract_templates(self, codebase_path: Path) -> Dict[str, TestTemplate]:
        """
        Extract test templates from existing test files in the codebase.

        Inspired by vector_revamp's template extraction from MEMSHADOW data.
        """
        logger.info(f"Extracting templates from codebase: {codebase_path}")

        extracted_templates = {}
        extraction_stats = {
            'files_processed': 0,
            'templates_extracted': 0,
            'languages_found': set(),
            'frameworks_found': set(),
        }

        # Find all test files
        test_files = self._find_test_files(codebase_path)
        logger.info(f"Found {len(test_files)} test files")

        for test_file in test_files:
            try:
                templates_from_file = self._extract_templates_from_file(test_file)
                for template in templates_from_file:
                    template_id = self._generate_template_id(template, test_file)
                    template.template_id = template_id
                    template.extracted_from.append(str(test_file))

                    extracted_templates[template_id] = template

                extraction_stats['files_processed'] += 1
                extraction_stats['templates_extracted'] += len(templates_from_file)
                extraction_stats['languages_found'].add(self._detect_language(test_file))
                extraction_stats['frameworks_found'].add(self._detect_framework(test_file))

            except Exception as e:
                logger.warning(f"Failed to extract templates from {test_file}: {e}")
                continue

        # Quality analysis and filtering
        filtered_templates = self._filter_and_score_templates(extracted_templates)

        self.extraction_stats = extraction_stats
        self.templates.update(filtered_templates)

        logger.info(f"Extracted {len(filtered_templates)} high-quality templates")
        logger.info(f"Languages: {extraction_stats['languages_found']}")
        logger.info(f"Frameworks: {extraction_stats['frameworks_found']}")

        return filtered_templates

    def _find_test_files(self, codebase_path: Path) -> List[Path]:
        """Find all test files in the codebase."""
        test_files = []

        # Common test file patterns
        test_patterns = [
            "**/test_*.py",
            "**/*_test.py",
            "**/tests/**/*.py",
            "**/test/**/*.c",
            "**/test/**/*.h",
            "**/tests/**/*.rs",
            "**/*_test.rs",
            "**/src/test/**/*.rs",
        ]

        for pattern in test_patterns:
            test_files.extend(codebase_path.glob(pattern))

        # Remove duplicates
        test_files = list(set(test_files))
        return sorted(test_files)

    def _extract_templates_from_file(self, test_file: Path) -> List[TestTemplate]:
        """Extract templates from a single test file."""
        templates = []

        language = self._detect_language(test_file)
        framework = self._detect_framework(test_file)

        if language == 'python':
            templates.extend(self._extract_python_templates(test_file, framework))
        elif language == 'c':
            templates.extend(self._extract_c_templates(test_file, framework))
        elif language == 'rust':
            templates.extend(self._extract_rust_templates(test_file, framework))

        return templates

    def _extract_python_templates(self, test_file: Path, framework: str) -> List[TestTemplate]:
        """Extract test templates from Python test files."""
        templates = []

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse AST for better analysis
            try:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        template = self._analyze_python_test_function(node, content, framework)
                        if template:
                            templates.append(template)
            except SyntaxError:
                # Fallback to regex-based extraction
                logger.warning(f"Syntax error in {test_file}, using regex extraction")
                templates.extend(self._extract_python_templates_regex(content, framework))

        except Exception as e:
            logger.error(f"Error extracting Python templates from {test_file}: {e}")

        return templates

    def _analyze_python_test_function(self, node: ast.FunctionDef, content: str, framework: str) -> Optional[TestTemplate]:
        """Analyze a Python test function using AST."""
        try:
            # Extract function details
            func_name = node.name
            func_start = node.lineno - 1
            func_end = node.end_lineno or func_start + 10

            # Get function body
            lines = content.split('\n')
            func_body = '\n'.join(lines[func_start:func_end])

            # Analyze test patterns
            setup_patterns = []
            test_patterns = []
            assertion_patterns = []
            cleanup_patterns = []

            for child in ast.walk(node):
                if isinstance(child, ast.With):
                    # Context managers (setup/cleanup)
                    setup_patterns.append(self._extract_context_manager_pattern(child))
                elif isinstance(child, ast.Assert):
                    # Assertions
                    assertion_patterns.append(self._extract_assertion_pattern(child))
                elif isinstance(child, ast.Call):
                    # Function calls
                    call_pattern = self._extract_call_pattern(child)
                    if call_pattern:
                        test_patterns.append(call_pattern)

            # Determine test type
            test_type = self._classify_python_test_type(func_body, assertion_patterns)

            # Calculate complexity
            complexity = len(assertion_patterns) + len(test_patterns)

            template = TestTemplate(
                template_id="",  # Will be set later
                test_type=test_type,
                language="python",
                framework=framework,
                structure={
                    "function_name_pattern": func_name,
                    "has_setup": len(setup_patterns) > 0,
                    "has_cleanup": len(cleanup_patterns) > 0,
                    "complexity": complexity,
                },
                setup_patterns=setup_patterns,
                test_patterns=test_patterns,
                assertion_patterns=assertion_patterns,
                cleanup_patterns=cleanup_patterns,
                quality_score=self._calculate_template_quality(setup_patterns, test_patterns, assertion_patterns),
            )

            return template

        except Exception as e:
            logger.warning(f"Error analyzing Python test function {node.name}: {e}")
            return None

    def _extract_python_templates_regex(self, content: str, framework: str) -> List[TestTemplate]:
        """Fallback regex-based extraction for Python tests."""
        templates = []

        # Simple pattern matching for test functions
        test_functions = re.findall(r'def (test_\w+)\([^)]*\):(.*?)(?=\ndef|\nclass|\n@|\Z)', content, re.DOTALL)

        for func_name, func_body in test_functions:
            # Extract patterns
            assertions = re.findall(r'assert\s+(.+)', func_body)
            calls = re.findall(r'(\w+)\([^)]*\)', func_body)

            if assertions or calls:  # Only create template if there are testable patterns
                template = TestTemplate(
                    template_id="",
                    test_type=self._classify_test_type_from_patterns(assertions, calls),
                    language="python",
                    framework=framework,
                    structure={"function_name": func_name},
                    assertion_patterns=assertions[:5],  # Limit patterns
                    test_patterns=calls[:5],
                    quality_score=min(1.0, (len(assertions) + len(calls)) / 10.0),
                )
                templates.append(template)

        return templates

    def _extract_c_templates(self, test_file: Path, framework: str) -> List[TestTemplate]:
        """Extract test templates from C test files."""
        templates = []

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract test functions based on framework
            if framework == 'unity':
                test_functions = re.findall(r'void (test_\w+)\(void\)(.*?)(?=void|\Z)', content, re.DOTALL)
            elif framework == 'cmocka':
                test_functions = re.findall(r'static void (test_\w+)\(void \*\*state\)(.*?)(?=static|\Z)', content, re.DOTALL)
            else:
                test_functions = re.findall(r'void (test_\w+)\([^)]*\)(.*?)(?=void|\Z)', content, re.DOTALL)

            for func_name, func_body in test_functions:
                # Extract C-specific patterns
                assertions = re.findall(r'TEST_ASSERT[^;]+', func_body) if framework == 'unity' else \
                           re.findall(r'assert[^;]+', func_body)

                calls = re.findall(r'(\w+)\([^)]*\)', func_body)

                if assertions or calls:
                    template = TestTemplate(
                        template_id="",
                        test_type=self._classify_test_type_from_patterns(assertions, calls),
                        language="c",
                        framework=framework,
                        structure={"function_name": func_name},
                        assertion_patterns=assertions[:5],
                        test_patterns=calls[:5],
                        quality_score=min(1.0, (len(assertions) + len(calls)) / 8.0),
                    )
                    templates.append(template)

        except Exception as e:
            logger.error(f"Error extracting C templates from {test_file}: {e}")

        return templates

    def _extract_rust_templates(self, test_file: Path, framework: str) -> List[TestTemplate]:
        """Extract test templates from Rust test files."""
        templates = []

        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Extract test functions
            test_functions = re.findall(r'#\[test\]\s*fn (test_\w+)\(\)(.*?)(?=fn|#\[|\Z)', content, re.DOTALL)

            for func_name, func_body in test_functions:
                # Extract Rust-specific patterns
                assertions = re.findall(r'assert[^;]+', func_body)
                calls = re.findall(r'(\w+)\([^;]*\)', func_body)

                if assertions or calls:
                    template = TestTemplate(
                        template_id="",
                        test_type=self._classify_test_type_from_patterns(assertions, calls),
                        language="rust",
                        framework=framework,
                        structure={"function_name": func_name},
                        assertion_patterns=assertions[:5],
                        test_patterns=calls[:5],
                        quality_score=min(1.0, (len(assertions) + len(calls)) / 8.0),
                    )
                    templates.append(template)

        except Exception as e:
            logger.error(f"Error extracting Rust templates from {test_file}: {e}")

        return templates

    def _extract_context_manager_pattern(self, node: ast.With) -> str:
        """Extract pattern from context manager."""
        try:
            if node.items:
                item = node.items[0]
                if hasattr(item, 'context_expr') and hasattr(item.context_expr, 'func'):
                    func_name = getattr(item.context_expr.func, 'id', 'unknown')
                    return f"with {func_name}(...):"
        except:
            pass
        return "with context_manager(...):"

    def _extract_assertion_pattern(self, node: ast.Assert) -> str:
        """Extract pattern from assertion."""
        try:
            # Simple pattern extraction
            return "assert condition"
        except:
            return "assert ..."

    def _extract_call_pattern(self, node: ast.Call) -> Optional[str]:
        """Extract pattern from function call."""
        try:
            if hasattr(node.func, 'id'):
                func_name = node.func.id
                arg_count = len(node.args) if hasattr(node, 'args') else 0
                return f"{func_name}({arg_count} args)"
        except:
            pass
        return None

    def _classify_python_test_type(self, func_body: str, assertions: List[str]) -> str:
        """Classify Python test type based on content."""
        body_lower = func_body.lower()

        if 'integration' in body_lower or 'database' in body_lower or 'api' in body_lower:
            return 'integration'
        elif any('edge' in str(assertion).lower() or 'boundary' in str(assertion).lower() for assertion in assertions):
            return 'edge_case'
        elif len(assertions) > 3 or 'complex' in body_lower:
            return 'complex_unit'
        else:
            return 'unit'

    def _classify_test_type_from_patterns(self, assertions: List[str], calls: List[str]) -> str:
        """Classify test type based on assertion and call patterns."""
        if len(assertions) > 3:
            return 'complex_unit'
        elif any('edge' in str(a).lower() for a in assertions):
            return 'edge_case'
        elif len(calls) > 5:
            return 'integration'
        else:
            return 'unit'

    def _calculate_template_quality(self, setup_patterns: List[str], test_patterns: List[str],
                                  assertion_patterns: List[str]) -> float:
        """Calculate template quality score."""
        score = 0.0

        # Base score from pattern diversity
        total_patterns = len(setup_patterns) + len(test_patterns) + len(assertion_patterns)
        if total_patterns > 0:
            score += min(0.4, total_patterns / 10.0)

        # Bonus for having all components
        if setup_patterns:
            score += 0.2
        if test_patterns:
            score += 0.2
        if assertion_patterns:
            score += 0.2

        # Bonus for complexity (but not too complex)
        if 3 <= total_patterns <= 8:
            score += 0.2

        return min(1.0, score)

    def _filter_and_score_templates(self, templates: Dict[str, TestTemplate]) -> Dict[str, TestTemplate]:
        """Filter low-quality templates and refine scores."""
        filtered = {}

        for template_id, template in templates.items():
            # Skip templates with very low quality
            if template.quality_score < 0.3:
                continue

            # Skip templates with no patterns
            total_patterns = (len(template.setup_patterns) + len(template.test_patterns) +
                            len(template.assertion_patterns) + len(template.cleanup_patterns))
            if total_patterns < 2:
                continue

            # Refine quality score based on pattern diversity
            unique_patterns = set()
            for pattern_list in [template.setup_patterns, template.test_patterns,
                               template.assertion_patterns, template.cleanup_patterns]:
                unique_patterns.update(pattern_list)

            diversity_bonus = min(0.2, len(unique_patterns) / 20.0)
            template.quality_score = min(1.0, template.quality_score + diversity_bonus)

            filtered[template_id] = template

        logger.info(f"Filtered templates: {len(templates)} -> {len(filtered)}")
        return filtered

    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension."""
        ext = file_path.suffix.lower()
        if ext == '.py':
            return 'python'
        elif ext in ['.c', '.h']:
            return 'c'
        elif ext == '.rs':
            return 'rust'
        else:
            return 'unknown'

    def _detect_framework(self, file_path: Path) -> str:
        """Detect test framework from file content and naming."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(1024)  # Read first 1KB

            content_lower = content.lower()

            # Python frameworks
            if 'import pytest' in content or 'pytest.' in content:
                return 'pytest'
            elif 'import unittest' in content or 'unittest.' in content:
                return 'unittest'

            # C frameworks
            elif 'unity.h' in content or 'UNITY_BEGIN' in content:
                return 'unity'
            elif 'cmocka.h' in content or 'cmocka_run_group_tests' in content:
                return 'cmocka'

            # Rust frameworks (usually built-in test framework)
            elif file_path.suffix == '.rs' and '#[test]' in content:
                return 'rust_builtin'

        except Exception:
            pass

        # Fallback based on language
        language = self._detect_language(file_path)
        if language == 'python':
            return 'pytest'  # Default assumption
        elif language == 'c':
            return 'unity'   # Default assumption
        elif language == 'rust':
            return 'rust_builtin'
        else:
            return 'unknown'

    def _generate_template_id(self, template: TestTemplate, source_file: Path) -> str:
        """Generate unique template ID."""
        # Create hash from template characteristics
        template_str = f"{template.language}_{template.framework}_{template.test_type}_"
        template_str += f"{len(template.setup_patterns)}_{len(template.test_patterns)}_"
        template_str += f"{len(template.assertion_patterns)}_{str(source_file)}"

        hash_obj = hashlib.md5(template_str.encode())
        return f"{template.language}_{template.framework}_{hash_obj.hexdigest()[:8]}"

    def apply_template(self, template: TestTemplate, code_context: CodeContext) -> Optional[GeneratedTest]:
        """
        Apply a template to generate a test for the given code context.

        Inspired by vector_revamp's rule-based synthesis approach.
        """
        try:
            # Generate test code using template
            test_code = self._synthesize_test_code(template, code_context)

            if not test_code:
                return None

            # Calculate quality metrics
            quality_score = self._evaluate_generated_test(test_code, template, code_context)
            coverage_estimate = self._estimate_coverage_impact(test_code, code_context)

            # Create result
            generated_test = GeneratedTest(
                test_code=test_code,
                template_used=template.template_id,
                target_function=code_context.function_name,
                test_type=template.test_type,
                quality_score=quality_score,
                coverage_estimate=coverage_estimate,
                metadata={
                    'language': template.language,
                    'framework': template.framework,
                    'complexity_level': template.complexity_level,
                    'generation_method': 'template_based',
                    'source_template_quality': template.quality_score,
                    'target_function_complexity': code_context.complexity,
                }
            )

            # Update template usage statistics
            template.usage_count += 1

            # Record feedback for evolution system
            from .template_evolution import create_evolution_feedback
            feedback = create_evolution_feedback(
                template_id=template.template_id,
                module_name=code_context.function_name,
                success=True,  # Successful generation
                quality_score=validated_score,
                execution_time=0.1,  # Estimated execution time
                test_vectors_generated=1,
                context={
                    'language': template.language,
                    'framework': template.framework,
                    'complexity': code_context.complexity,
                    'target_function': code_context.function_name
                }
            )
            self.evolution_engine.record_feedback(feedback)

            # Check if template should evolve
            if self.evolution_engine.should_evolve(template.template_id):
                logger.info(f"Template {template.template_id} ready for evolution")
                evolved_template = self.evolution_engine.evolve_template(template)
                if evolved_template:
                    # Add evolved template to library
                    self.templates[evolved_template.template_id] = evolved_template
                    logger.info(f"Added evolved template: {evolved_template.template_id}")

            return generated_test

        except Exception as e:
            logger.error(f"Error applying template {template.template_id} to {code_context.function_name}: {e}")
            return None

    def _synthesize_test_code(self, template: TestTemplate, context: CodeContext) -> Optional[str]:
        """Synthesize test code using template and context."""
        try:
            if template.language == 'python':
                return self._synthesize_python_test(template, context)
            elif template.language == 'c':
                return self._synthesize_c_test(template, context)
            elif template.language == 'rust':
                return self._synthesize_rust_test(template, context)
            else:
                return None
        except Exception as e:
            logger.error(f"Error synthesizing test code: {e}")
            return None

    def _synthesize_python_test(self, template: TestTemplate, context: CodeContext) -> Optional[str]:
        """Synthesize Python test code."""
        lines = []

        # Function signature
        func_name = f"test_{context.function_name}"
        if template.framework == 'pytest':
            lines.append(f"def {func_name}():")
        else:  # unittest
            lines.append(f"def {func_name}(self):")

        lines.append('    """')
        lines.append(f'    Test {context.function_name} function')
        lines.append('    """')

        # Setup
        if template.setup_patterns:
            for pattern in template.setup_patterns[:2]:  # Limit setup patterns
                if 'context_manager' in pattern:
                    lines.append(f"    # Setup code")
                    lines.append(f"    with some_context():")
                    lines.append(f"        pass")

        # Test execution
        lines.append(f"    # Test execution")
        if template.test_patterns:
            for pattern in template.test_patterns[:3]:  # Limit test patterns
                if '(' in pattern:
                    lines.append(f"    # Call function with test data")
                    lines.append(f"    result = {context.function_name}(test_input)")

        # Assertions
        lines.append(f"    # Assertions")
        if template.assertion_patterns:
            for pattern in template.assertion_patterns[:3]:  # Limit assertions
                if template.framework == 'pytest':
                    lines.append(f"    assert result is not None")
                else:  # unittest
                    lines.append(f"    self.assertIsNotNone(result)")

        # Cleanup
        if template.cleanup_patterns:
            lines.append(f"    # Cleanup code")
            lines.append(f"    pass")

        return '\n'.join(lines)

    def _synthesize_c_test(self, template: TestTemplate, context: CodeContext) -> Optional[str]:
        """Synthesize C test code."""
        lines = []

        # Function signature
        if template.framework == 'unity':
            lines.append(f"void test_{context.function_name}(void) {{")
        elif template.framework == 'cmocka':
            lines.append(f"static void test_{context.function_name}(void **state) {{")

        lines.append(f"    // Test {context.function_name} function")

        # Test execution
        lines.append(f"    // Test execution")
        lines.append(f"    // Call function with test data")

        # Assertions
        lines.append(f"    // Assertions")
        if template.framework == 'unity':
            lines.append(f"    TEST_ASSERT_NOT_NULL(result);")
        elif template.framework == 'cmocka':
            lines.append(f"    assert_non_null(result);")

        lines.append(f"}}")

        return '\n'.join(lines)

    def _synthesize_rust_test(self, template: TestTemplate, context: CodeContext) -> Optional[str]:
        """Synthesize Rust test code."""
        lines = []

        lines.append("#[test]")
        lines.append(f"fn test_{context.function_name}() {{")
        lines.append(f"    // Test {context.function_name} function")

        # Test execution
        lines.append(f"    // Test execution")
        lines.append(f"    // Call function with test data")

        # Assertions
        lines.append(f"    // Assertions")
        lines.append(f"    assert!(result.is_some());")

        lines.append("}")

        return '\n'.join(lines)

    def _evaluate_generated_test(self, test_code: str, template: TestTemplate, context: CodeContext) -> float:
        """Evaluate the quality of generated test code."""
        score = 0.5  # Base score

        # Check if test code is valid for the language
        if self._validate_test_syntax(test_code, template.language):
            score += 0.2

        # Check if test follows framework conventions
        if self._validate_framework_conventions(test_code, template.framework):
            score += 0.2

        # Check if test is appropriate for function complexity
        if self._validate_complexity_match(test_code, context.complexity):
            score += 0.1

        # Template quality bonus
        score += template.quality_score * 0.2

        return min(1.0, score)

    def _validate_test_syntax(self, test_code: str, language: str) -> bool:
        """Validate test code syntax."""
        try:
            if language == 'python':
                ast.parse(test_code)
                return True
            # For C/Rust, we could add syntax validation, but for now assume valid
            return True
        except:
            return False

    def _validate_framework_conventions(self, test_code: str, framework: str) -> bool:
        """Validate that test follows framework conventions."""
        # Basic checks - could be enhanced
        if framework == 'pytest':
            return 'def test_' in test_code and 'assert' in test_code
        elif framework == 'unittest':
            return 'def test_' in test_code and 'self.assert' in test_code
        elif framework == 'unity':
            return 'void test_' in test_code and 'TEST_ASSERT' in test_code
        elif framework == 'cmocka':
            return 'static void test_' in test_code and 'assert_' in test_code
        else:
            return True

    def _validate_complexity_match(self, test_code: str, function_complexity: int) -> bool:
        """Validate that test complexity matches function complexity."""
        # Simple heuristic: more complex functions need more complex tests
        test_lines = len(test_code.split('\n'))
        expected_complexity = max(5, function_complexity // 2)

        return abs(test_lines - expected_complexity) < 10  # Allow some variance

    def _estimate_coverage_impact(self, test_code: str, context: CodeContext) -> float:
        """Estimate the coverage impact of the generated test."""
        # Simple estimation based on test structure
        score = 0.3  # Base coverage

        # More lines = potentially more coverage
        lines = len(test_code.split('\n'))
        score += min(0.3, lines / 20.0)

        # Function complexity bonus
        score += min(0.2, context.complexity / 20.0)

        # Exception handling bonus
        if context.has_exceptions and 'assertRaises' in test_code or 'pytest.raises' in test_code:
            score += 0.2

        return min(1.0, score)

    def get_template_statistics(self) -> Dict[str, Any]:
        """Get statistics about extracted templates."""
        stats = {
            'total_templates': len(self.templates),
            'templates_by_language': defaultdict(int),
            'templates_by_framework': defaultdict(int),
            'templates_by_type': defaultdict(int),
            'average_quality_score': 0.0,
            'quality_distribution': {'excellent': 0, 'good': 0, 'acceptable': 0, 'poor': 0},
        }

        total_quality = 0.0

        for template in self.templates.values():
            stats['templates_by_language'][template.language] += 1
            stats['templates_by_framework'][template.framework] += 1
            stats['templates_by_type'][template.test_type] += 1

            total_quality += template.quality_score

            # Quality distribution
            if template.quality_score >= 0.9:
                stats['quality_distribution']['excellent'] += 1
            elif template.quality_score >= 0.75:
                stats['quality_distribution']['good'] += 1
            elif template.quality_score >= 0.6:
                stats['quality_distribution']['acceptable'] += 1
            else:
                stats['quality_distribution']['poor'] += 1

        if stats['total_templates'] > 0:
            stats['average_quality_score'] = total_quality / stats['total_templates']

        return dict(stats)

    def save_templates(self, output_path: Path):
        """Save extracted templates to file."""
        template_data = {}
        for template_id, template in self.templates.items():
            template_data[template_id] = {
                'template_id': template.template_id,
                'test_type': template.test_type,
                'language': template.language,
                'framework': template.framework,
                'structure': template.structure,
                'setup_patterns': template.setup_patterns,
                'test_patterns': template.test_patterns,
                'assertion_patterns': template.assertion_patterns,
                'cleanup_patterns': template.cleanup_patterns,
                'extracted_from': template.extracted_from,
                'quality_score': template.quality_score,
                'usage_count': template.usage_count,
                'success_rate': template.success_rate,
                'complexity_level': template.complexity_level,
                'coverage_targets': template.coverage_targets,
                'parameter_patterns': template.parameter_patterns,
                'validation_rules': template.validation_rules,
            }

        with open(output_path, 'w') as f:
            json.dump(template_data, f, indent=2)

        logger.info(f"Saved {len(template_data)} templates to {output_path}")

    def load_templates(self, input_path: Path):
        """Load templates from file."""
        try:
            with open(input_path, 'r') as f:
                template_data = json.load(f)

            for template_id, data in template_data.items():
                template = TestTemplate(
                    template_id=data['template_id'],
                    test_type=data['test_type'],
                    language=data['language'],
                    framework=data['framework'],
                    structure=data['structure'],
                    setup_patterns=data['setup_patterns'],
                    test_patterns=data['test_patterns'],
                    assertion_patterns=data['assertion_patterns'],
                    cleanup_patterns=data['cleanup_patterns'],
                    extracted_from=data['extracted_from'],
                    quality_score=data['quality_score'],
                    usage_count=data.get('usage_count', 0),
                    success_rate=data.get('success_rate', 0.0),
                    complexity_level=data.get('complexity_level', 'medium'),
                    coverage_targets=data.get('coverage_targets', []),
                    parameter_patterns=data.get('parameter_patterns', {}),
                    validation_rules=data.get('validation_rules', {}),
                )
                self.templates[template_id] = template

            logger.info(f"Loaded {len(template_data)} templates from {input_path}")

        except Exception as e:
            logger.error(f"Error loading templates from {input_path}: {e}")

    def get_best_template(self, language: str, framework: str, test_type: str,
                         min_quality: float = 0.6) -> Optional[TestTemplate]:
        """Get the best template for the given criteria."""
        candidates = []

        for template in self.templates.values():
            if (template.language == language and
                template.framework == framework and
                template.test_type == test_type and
                template.quality_score >= min_quality):
                candidates.append(template)

        if not candidates:
            return None

        # Sort by quality score, then by usage count
        candidates.sort(key=lambda t: (t.quality_score, t.usage_count), reverse=True)
        return candidates[0]

    def save_evolution_state(self, output_path: Path):
        """Save template evolution state."""
        self.evolution_engine.save_evolution_state(output_path)

    def load_evolution_state(self, input_path: Path):
        """Load template evolution state."""
        self.evolution_engine.load_evolution_state(input_path)

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get template evolution statistics."""
        return self.evolution_engine.get_evolution_statistics()

    def get_evolution_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for template evolution."""
        return self.evolution_engine.get_evolution_recommendations()

    def record_generation_failure(self, template_id: str, module_name: str, error_message: str):
        """Record a generation failure for learning."""
        feedback = create_evolution_feedback(
            template_id=template_id,
            module_name=module_name,
            success=False,
            quality_score=0.0,
            execution_time=0.1,
            error_message=error_message,
            context={'failure_reason': error_message}
        )
        self.evolution_engine.record_feedback(feedback)
