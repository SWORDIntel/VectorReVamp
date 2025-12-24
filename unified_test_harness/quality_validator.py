"""
Quality Validation System for VectorReVamp Test Framework

Comprehensive test quality validation inspired by vector_revamp's
multi-layer validation approach with structural, functional, statistical,
and domain-specific checks.
"""

import ast
import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a quality validation check."""
    check_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    severity: str  # 'critical', 'high', 'medium', 'low'
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class QualityReport:
    """Comprehensive quality report for generated tests."""
    overall_score: float
    validation_results: List[ValidationResult]
    quality_distribution: Dict[str, int]  # severity -> count
    critical_issues: int
    high_issues: int
    recommendations: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class TestQualityValidator:
    """
    Comprehensive test quality validator inspired by vector_revamp's
    multi-layer validation approach.
    """

    def __init__(self):
        self.validation_checks = {
            'structural': self._validate_structural,
            'functional': self._validate_functional,
            'syntactic': self._validate_syntactic,
            'coverage': self._validate_coverage_potential,
            'complexity': self._validate_complexity_match,
            'framework': self._validate_framework_compliance,
            'domain': self._validate_domain_specific,
            'statistical': self._validate_statistical_properties,
        }

        # Language-specific validation rules
        self.language_rules = {
            'python': self._get_python_validation_rules(),
            'c': self._get_c_validation_rules(),
            'rust': self._get_rust_validation_rules(),
        }

        # Framework-specific validation rules
        self.framework_rules = {
            'pytest': self._get_pytest_validation_rules(),
            'unittest': self._get_unittest_validation_rules(),
            'cmocka': self._get_cmocka_validation_rules(),
            'unity': self._get_unity_validation_rules(),
            'rust_builtin': self._get_rust_validation_rules(),
        }

    def validate_test(self, test_code: str, language: str, framework: str,
                     target_function: str = "", context: Dict[str, Any] = None) -> QualityReport:
        """
        Perform comprehensive quality validation on generated test code.

        Args:
            test_code: The generated test code
            language: Programming language ('python', 'c', 'rust')
            framework: Test framework ('pytest', 'unittest', etc.)
            target_function: Name of function being tested
            context: Additional context information

        Returns:
            Comprehensive quality report
        """
        if context is None:
            context = {}

        validation_results = []
        total_score = 0.0
        weights = {
            'structural': 0.25,
            'functional': 0.25,
            'syntactic': 0.15,
            'coverage': 0.15,
            'complexity': 0.10,
            'framework': 0.05,
            'domain': 0.03,
            'statistical': 0.02,
        }

        # Run all validation checks
        for check_name, check_func in self.validation_checks.items():
            try:
                result = check_func(test_code, language, framework, target_function, context)
                validation_results.append(result)
                total_score += result.score * weights.get(check_name, 0.1)
            except Exception as e:
                logger.warning(f"Validation check '{check_name}' failed: {e}")
                # Add failed check result
                failed_result = ValidationResult(
                    check_name=check_name,
                    passed=False,
                    score=0.0,
                    severity='high',
                    message=f"Validation check failed: {e}",
                    details={'error': str(e)}
                )
                validation_results.append(failed_result)

        # Calculate quality distribution
        quality_distribution = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        critical_issues = 0
        high_issues = 0

        recommendations = []
        for result in validation_results:
            quality_distribution[result.severity] += 1
            if result.severity == 'critical':
                critical_issues += 1
            elif result.severity == 'high':
                high_issues += 1
            recommendations.extend(result.recommendations)

        # Remove duplicates and sort recommendations
        recommendations = list(set(recommendations))

        # Create comprehensive report
        report = QualityReport(
            overall_score=min(1.0, total_score),
            validation_results=validation_results,
            quality_distribution=quality_distribution,
            critical_issues=critical_issues,
            high_issues=high_issues,
            recommendations=recommendations,
            metadata={
                'language': language,
                'framework': framework,
                'target_function': target_function,
                'test_code_length': len(test_code),
                'validation_checks_run': len(validation_results),
                'validation_timestamp': context.get('timestamp', None),
            }
        )

        return report

    def _validate_structural(self, test_code: str, language: str, framework: str,
                           target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate structural integrity of test code."""
        score = 1.0
        issues = []

        try:
            # Check for basic structural elements
            if language == 'python':
                # Must have at least one test function
                if not re.search(r'def test_', test_code):
                    score -= 0.5
                    issues.append("No test function found")

                # Check for proper indentation (basic)
                lines = test_code.split('\n')
                indent_issues = 0
                for line in lines:
                    if line.strip() and not line.startswith(' ') and not line.startswith('\t') and ':' not in line:
                        # Non-indented line that's not a function/class def
                        if not any(line.startswith(x) for x in ['def ', 'class ', 'import ', 'from ', 'if ', 'for ', 'with ']):
                            indent_issues += 1

                if indent_issues > 0:
                    score -= min(0.3, indent_issues * 0.1)
                    issues.append(f"Indentation issues found: {indent_issues}")

            elif language in ['c', 'rust']:
                # Check for basic function structure
                if not re.search(r'\w+\s+\w+\s*\([^)]*\)\s*\{', test_code):
                    score -= 0.4
                    issues.append("No function definition found")

            # Check for reasonable length
            lines = len(test_code.split('\n'))
            if lines < 3:
                score -= 0.3
                issues.append("Test too short")
            elif lines > 100:
                score -= 0.1
                issues.append("Test unusually long")

        except Exception as e:
            score = 0.0
            issues.append(f"Structural validation failed: {e}")

        severity = 'critical' if score < 0.5 else 'medium' if score < 0.8 else 'low'

        return ValidationResult(
            check_name='structural',
            passed=score >= 0.7,
            score=score,
            severity=severity,
            message=f"Structural validation: {score:.2f}/1.0",
            details={'issues_found': issues},
            recommendations=["Fix structural issues: " + ", ".join(issues)] if issues else []
        )

    def _validate_functional(self, test_code: str, language: str, framework: str,
                           target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate functional aspects of test code."""
        score = 1.0
        issues = []

        try:
            # Check if test actually calls the target function
            if target_function:
                if target_function not in test_code:
                    score -= 0.6
                    issues.append(f"Target function '{target_function}' not called")

            # Check for assertions (framework-specific)
            has_assertions = False

            if framework == 'pytest':
                has_assertions = 'assert' in test_code
            elif framework == 'unittest':
                has_assertions = 'self.assert' in test_code
            elif framework in ['unity', 'cmocka']:
                has_assertions = 'TEST_ASSERT' in test_code or 'assert_' in test_code
            elif framework == 'rust_builtin':
                has_assertions = 'assert!' in test_code or 'assert_eq!' in test_code

            if not has_assertions:
                score -= 0.5
                issues.append("No assertions found")

            # Check for meaningful test logic
            # Look for variable assignments, function calls, control flow
            meaningful_elements = 0
            if '=' in test_code and '==' not in test_code.split('=')[0]:  # Assignment, not comparison
                meaningful_elements += 1
            if any(keyword in test_code for keyword in ['if ', 'for ', 'while ', 'with ', 'try:', 'except:']):
                meaningful_elements += 1
            if re.search(r'\w+\([^)]*\)', test_code):  # Function calls
                meaningful_elements += 1

            if meaningful_elements < 2:
                score -= 0.3
                issues.append("Limited test logic")

        except Exception as e:
            score = 0.0
            issues.append(f"Functional validation failed: {e}")

        severity = 'high' if score < 0.6 else 'medium' if score < 0.8 else 'low'

        return ValidationResult(
            check_name='functional',
            passed=score >= 0.7,
            score=score,
            severity=severity,
            message=f"Functional validation: {score:.2f}/1.0",
            details={'issues_found': issues, 'has_assertions': 'assertions' in test_code},
            recommendations=["Add functional elements: " + ", ".join(issues)] if issues else []
        )

    def _validate_syntactic(self, test_code: str, language: str, framework: str,
                          target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate syntax correctness."""
        score = 1.0
        issues = []

        try:
            if language == 'python':
                try:
                    ast.parse(test_code)
                except SyntaxError as e:
                    score = 0.0
                    issues.append(f"Syntax error: {e}")
                except Exception as e:
                    score -= 0.5
                    issues.append(f"Parse error: {e}")

            # Check for common syntax issues
            # Unmatched brackets
            if test_code.count('(') != test_code.count(')'):
                score -= 0.3
                issues.append("Unmatched parentheses")

            if test_code.count('[') != test_code.count(']'):
                score -= 0.3
                issues.append("Unmatched brackets")

            if test_code.count('{') != test_code.count('}'):
                score -= 0.3
                issues.append("Unmatched braces")

            # Check for incomplete statements
            lines = test_code.split('\n')
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.endswith((':', ';', ',', '{', '}', ')', ']', '"', "'")):
                    if not any(stripped.startswith(x) for x in ['#', 'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except:', 'with ']):
                        # This might be incomplete - check next line
                        if i + 1 < len(lines) and not lines[i + 1].strip().startswith(' '):
                            score -= 0.1
                            if len(issues) < 3:  # Limit issue count
                                issues.append(f"Possible incomplete statement at line {i+1}")

        except Exception as e:
            score = 0.0
            issues.append(f"Syntactic validation failed: {e}")

        severity = 'critical' if score < 0.3 else 'high' if score < 0.7 else 'low'

        return ValidationResult(
            check_name='syntactic',
            passed=score >= 0.8,
            score=score,
            severity=severity,
            message=f"Syntactic validation: {score:.2f}/1.0",
            details={'issues_found': issues},
            recommendations=["Fix syntax issues: " + ", ".join(issues[:3])] if issues else []
        )

    def _validate_coverage_potential(self, test_code: str, language: str, framework: str,
                                   target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate potential coverage impact."""
        score = 0.5  # Base score
        issues = []

        try:
            # Estimate coverage based on test structure
            lines = len(test_code.split('\n'))

            # More lines generally mean more potential coverage
            if lines > 20:
                score += 0.3
            elif lines > 10:
                score += 0.2
            elif lines < 5:
                score -= 0.2
                issues.append("Test too short for meaningful coverage")

            # Check for multiple execution paths
            path_indicators = ['if ', 'elif ', 'else:', 'for ', 'while ', 'try:', 'except:']
            path_count = sum(1 for indicator in path_indicators if indicator in test_code)

            if path_count > 2:
                score += 0.2
            elif path_count == 0:
                score -= 0.1
                issues.append("No conditional logic")

            # Check for function calls (indicates interaction)
            call_count = len(re.findall(r'\w+\s*\([^)]*\)', test_code))
            if call_count > 3:
                score += 0.2
            elif call_count < 1:
                score -= 0.3
                issues.append("Few function calls")

            # Check for target function coverage
            if target_function and target_function in test_code:
                score += 0.2
            else:
                issues.append("Target function not directly tested")

        except Exception as e:
            score = 0.0
            issues.append(f"Coverage validation failed: {e}")

        severity = 'low'  # Coverage potential is not critical

        return ValidationResult(
            check_name='coverage',
            passed=score >= 0.6,
            score=min(1.0, score),
            severity=severity,
            message=f"Coverage potential: {score:.2f}/1.0",
            details={'estimated_lines': len(test_code.split('\n')), 'issues_found': issues},
            recommendations=["Improve coverage: " + ", ".join(issues)] if issues else []
        )

    def _validate_complexity_match(self, test_code: str, language: str, framework: str,
                                 target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate that test complexity matches target function complexity."""
        score = 0.8  # Default good score
        issues = []

        try:
            target_complexity = context.get('target_complexity', 1)
            test_lines = len(test_code.split('\n'))

            # Estimate test complexity based on structure
            test_complexity = 1
            if 'if ' in test_code:
                test_complexity += 1
            if 'for ' in test_code or 'while ' in test_code:
                test_complexity += 1
            if 'try:' in test_code:
                test_complexity += 1
            if len(re.findall(r'\w+\s*\([^)]*\)', test_code)) > 5:
                test_complexity += 1

            # Compare complexities
            complexity_ratio = test_complexity / max(1, target_complexity)

            if complexity_ratio < 0.5:
                score -= 0.3
                issues.append("Test too simple for target function complexity")
            elif complexity_ratio > 3.0:
                score -= 0.2
                issues.append("Test overly complex")

            # Length appropriateness
            expected_min_lines = target_complexity * 2
            expected_max_lines = target_complexity * 10

            if test_lines < expected_min_lines:
                score -= 0.2
                issues.append("Test too short for function complexity")
            elif test_lines > expected_max_lines:
                score -= 0.1
                issues.append("Test length disproportionate to function")

        except Exception as e:
            score = 0.5
            issues.append(f"Complexity validation failed: {e}")

        severity = 'medium' if score < 0.6 else 'low'

        return ValidationResult(
            check_name='complexity',
            passed=score >= 0.6,
            score=score,
            severity=severity,
            message=f"Complexity match: {score:.2f}/1.0",
            details={'issues_found': issues},
            recommendations=["Adjust complexity: " + ", ".join(issues)] if issues else []
        )

    def _validate_framework_compliance(self, test_code: str, language: str, framework: str,
                                      target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate framework-specific compliance."""
        score = 1.0
        issues = []

        try:
            rules = self.framework_rules.get(framework, {})
            if not rules:
                return ValidationResult(
                    check_name='framework',
                    passed=True,
                    score=0.8,  # Neutral score for unknown frameworks
                    severity='low',
                    message="Framework validation skipped (unknown framework)",
                    details={'framework': framework}
                )

            # Check required patterns
            for pattern_name, pattern in rules.get('required_patterns', {}).items():
                if not re.search(pattern, test_code):
                    score -= rules.get('pattern_weight', 0.2)
                    issues.append(f"Missing {pattern_name}")

            # Check forbidden patterns
            for pattern_name, pattern in rules.get('forbidden_patterns', {}).items():
                if re.search(pattern, test_code):
                    score -= rules.get('forbidden_weight', 0.3)
                    issues.append(f"Forbidden {pattern_name}")

        except Exception as e:
            score = 0.5
            issues.append(f"Framework validation failed: {e}")

        severity = 'high' if score < 0.7 else 'medium' if score < 0.9 else 'low'

        return ValidationResult(
            check_name='framework',
            passed=score >= 0.8,
            score=score,
            severity=severity,
            message=f"Framework compliance: {score:.2f}/1.0",
            details={'framework': framework, 'issues_found': issues},
            recommendations=["Fix framework issues: " + ", ".join(issues)] if issues else []
        )

    def _validate_domain_specific(self, test_code: str, language: str, framework: str,
                                target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate domain-specific rules."""
        score = 0.9  # Generally good unless issues found
        issues = []

        try:
            rules = self.language_rules.get(language, {})
            if not rules:
                return ValidationResult(
                    check_name='domain',
                    passed=True,
                    score=0.8,
                    severity='low',
                    message="Domain validation skipped (unknown language)",
                    details={'language': language}
                )

            # Check naming conventions
            for convention_name, pattern in rules.get('naming_conventions', {}).items():
                if not re.search(pattern, test_code):
                    score -= 0.1
                    issues.append(f"Poor {convention_name} naming")

            # Check code style
            for style_name, check_func in rules.get('style_checks', {}).items():
                if check_func and not check_func(test_code):
                    score -= 0.05
                    issues.append(f"Style issue: {style_name}")

        except Exception as e:
            score = 0.7
            issues.append(f"Domain validation failed: {e}")

        severity = 'low'  # Domain issues are not critical

        return ValidationResult(
            check_name='domain',
            passed=score >= 0.8,
            score=score,
            severity=severity,
            message=f"Domain validation: {score:.2f}/1.0",
            details={'language': language, 'issues_found': issues},
            recommendations=["Address domain issues: " + ", ".join(issues)] if issues else []
        )

    def _validate_statistical_properties(self, test_code: str, language: str, framework: str,
                                       target_function: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate statistical properties of the test."""
        score = 0.8  # Generally good
        issues = []

        try:
            # Analyze code patterns statistically
            lines = test_code.split('\n')
            avg_line_length = sum(len(line) for line in lines) / len(lines) if lines else 0

            # Check line length distribution
            if avg_line_length > 120:
                score -= 0.1
                issues.append("Lines too long")
            elif avg_line_length < 20:
                score -= 0.05
                issues.append("Lines too short")

            # Check comment ratio
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            comment_ratio = comment_lines / len(lines) if lines else 0

            if comment_ratio > 0.5:
                score -= 0.1
                issues.append("Too many comments")
            elif comment_ratio < 0.05:
                score -= 0.05
                issues.append("Too few comments")

            # Check token diversity (simplified)
            words = re.findall(r'\b\w+\b', test_code.lower())
            unique_words = len(set(words))
            total_words = len(words)
            diversity_ratio = unique_words / total_words if total_words > 0 else 0

            if diversity_ratio < 0.3:
                score -= 0.1
                issues.append("Low token diversity")

        except Exception as e:
            score = 0.6
            issues.append(f"Statistical validation failed: {e}")

        severity = 'low'  # Statistical issues are not critical

        return ValidationResult(
            check_name='statistical',
            passed=score >= 0.7,
            score=score,
            severity=severity,
            message=f"Statistical properties: {score:.2f}/1.0",
            details={'issues_found': issues},
            recommendations=["Improve statistics: " + ", ".join(issues)] if issues else []
        )

    # Language-specific validation rules
    def _get_python_validation_rules(self):
        return {
            'naming_conventions': {
                'functions': r'def (test_\w+|setUp|tearDown)',
                'variables': r'\b[a-z_][a-z0-9_]*\b'
            },
            'style_checks': {
                'no_tabs': lambda code: '\t' not in code,
                'reasonable_indentation': lambda code: not any(len(line) - len(line.lstrip()) > 16 for line in code.split('\n'))
            }
        }

    def _get_c_validation_rules(self):
        return {
            'naming_conventions': {
                'functions': r'(void|int|char)\s+(test_\w+|setUp|tearDown)',
                'variables': r'\b[a-z_][a-z0-9_]*\b'
            },
            'style_checks': {
                'braces': lambda code: self._check_c_braces(code)
            }
        }

    def _get_rust_validation_rules(self):
        return {
            'naming_conventions': {
                'functions': r'fn (test_\w+|setup|teardown)',
                'variables': r'\b[a-z_][a-z0-9_]*\b'
            },
            'style_checks': {
                'semicolons': lambda code: code.count(';') >= code.count('fn ')  # Basic check
            }
        }

    # Framework-specific validation rules
    def _get_pytest_validation_rules(self):
        return {
            'required_patterns': {
                'test_function': r'def test_',
                'assertions': r'assert\s+'
            },
            'pattern_weight': 0.3
        }

    def _get_unittest_validation_rules(self):
        return {
            'required_patterns': {
                'test_method': r'def test_',
                'assertions': r'self\.assert'
            },
            'pattern_weight': 0.3
        }

    def _get_cmocka_validation_rules(self):
        return {
            'required_patterns': {
                'test_function': r'static void test_',
                'assertions': r'assert_'
            },
            'pattern_weight': 0.3
        }

    def _get_unity_validation_rules(self):
        return {
            'required_patterns': {
                'test_function': r'void test_',
                'assertions': r'TEST_ASSERT'
            },
            'pattern_weight': 0.3
        }

    def _check_c_braces(self, code: str) -> bool:
        """Check C-style brace placement."""
        # Very basic check - could be enhanced
        lines = code.split('\n')
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('{') or stripped.startswith('}'):
                return True  # Found some bracing
        return True  # Allow no braces for simple functions

    def get_quality_summary(self, reports: List[QualityReport]) -> Dict[str, Any]:
        """Generate summary statistics across multiple quality reports."""
        if not reports:
            return {}

        total_reports = len(reports)
        avg_score = sum(r.overall_score for r in reports) / total_reports

        # Aggregate issues by severity
        total_issues = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        for report in reports:
            for severity, count in report.quality_distribution.items():
                total_issues[severity] += count

        # Collect common recommendations
        all_recommendations = []
        for report in reports:
            all_recommendations.extend(report.recommendations)

        # Get most common recommendations
        from collections import Counter
        common_recs = Counter(all_recommendations).most_common(5)

        return {
            'total_reports': total_reports,
            'average_score': avg_score,
            'quality_distribution': {
                'excellent': sum(1 for r in reports if r.overall_score >= 0.9),
                'good': sum(1 for r in reports if 0.8 <= r.overall_score < 0.9),
                'acceptable': sum(1 for r in reports if 0.7 <= r.overall_score < 0.8),
                'needs_improvement': sum(1 for r in reports if r.overall_score < 0.7)
            },
            'total_issues': total_issues,
            'common_recommendations': [rec[0] for rec in common_recs],
            'worst_performing_checks': self._get_worst_performing_checks(reports)
        }

    def _get_worst_performing_checks(self, reports: List[QualityReport]) -> List[Tuple[str, float]]:
        """Get the worst performing validation checks across all reports."""
        check_scores = {}

        for report in reports:
            for result in report.validation_results:
                if result.check_name not in check_scores:
                    check_scores[result.check_name] = []
                check_scores[result.check_name].append(result.score)

        # Calculate average scores
        avg_scores = {}
        for check_name, scores in check_scores.items():
            avg_scores[check_name] = sum(scores) / len(scores)

        # Return worst performing (lowest scores) first
        return sorted(avg_scores.items(), key=lambda x: x[1])[:5]
