"""
Intelligent Coverage Optimization & Strategic Planning for VectorReVamp

Advanced coverage analysis system that goes beyond basic gap identification
to provide strategic planning, risk assessment, and intelligent prioritization
of test generation efforts.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set, NamedTuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import statistics
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    nx = None
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


@dataclass
class CoverageGap:
    """Detailed coverage gap analysis."""
    module_name: str
    file_path: str
    function_name: str
    coverage_percentage: float
    lines_uncovered: int
    branches_uncovered: int
    risk_level: str  # 'critical', 'high', 'medium', 'low'
    business_impact: str  # 'high', 'medium', 'low'
    technical_complexity: str  # 'high', 'medium', 'low'
    dependencies: List[str] = field(default_factory=list)
    test_priority_score: float = 0.0
    estimated_test_effort: int = 1  # hours
    existing_tests: List[str] = field(default_factory=list)


@dataclass
class CoveragePrediction:
    """Prediction of coverage impact from test generation."""
    test_vector_id: str
    expected_coverage_increase: float
    confidence_score: float  # 0.0 to 1.0
    risk_assessment: str
    dependencies_satisfied: bool
    resource_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationStrategy:
    """Strategic plan for coverage optimization."""
    strategy_id: str
    target_modules: List[str]
    priority_order: List[str]
    estimated_completion_time: int  # hours
    expected_coverage_gain: float
    risk_mitigation_plan: Dict[str, Any]
    resource_requirements: Dict[str, Any]
    success_probability: float


class CoverageAnalysisEngine:
    """
    Advanced coverage analysis engine with strategic planning capabilities.

    Provides intelligent gap analysis, risk assessment, and optimization
    planning for test generation campaigns.
    """

    def __init__(self, config):
        self.config = config

        # Analysis state
        self.coverage_history: List[Dict[str, Any]] = []
        self.gap_analysis_cache: Dict[str, CoverageGap] = {}
        self.dependency_graph = nx.DiGraph() if HAS_NETWORKX else None

        # Strategic planning parameters
        self.strategy_params = {
            'risk_weight': 0.4,
            'impact_weight': 0.3,
            'complexity_weight': 0.2,
            'effort_weight': 0.1,
            'min_priority_score': 0.6,
            'max_strategy_modules': 20,
            'coverage_target_threshold': 0.85,
        }

        logger.info("CoverageAnalysisEngine initialized")

    def analyze_coverage_gaps(self, coverage_data: Dict[str, Any]) -> List[CoverageGap]:
        """
        Perform comprehensive coverage gap analysis with strategic insights.

        Args:
            coverage_data: Raw coverage data from coverage analyzer

        Returns:
            List of detailed coverage gaps with strategic analysis
        """
        gaps = []

        try:
            # Extract files and modules
            files_data = coverage_data.get('files', {})

            for file_path, file_coverage in files_data.items():
                module_name = self._extract_module_name(file_path)
                file_percentage = file_coverage.get('percentage', 0)

                # Analyze individual functions
                for func_name, func_data in file_coverage.get('functions', {}).items():
                    gap = self._analyze_function_gap(
                        module_name, file_path, func_name, func_data, file_percentage
                    )

                    if gap:
                        gaps.append(gap)
                        self.gap_analysis_cache[f"{module_name}.{func_name}"] = gap

            # Build dependency relationships
            self._build_dependency_graph(gaps)

            # Calculate strategic priorities
            self._calculate_strategic_priorities(gaps)

            # Sort by strategic priority
            gaps.sort(key=lambda g: g.test_priority_score, reverse=True)

            logger.info(f"Analyzed {len(gaps)} coverage gaps with strategic prioritization")

            return gaps

        except Exception as e:
            logger.error(f"Coverage gap analysis failed: {e}")
            return []

    def _analyze_function_gap(self, module_name: str, file_path: str, func_name: str,
                            func_data: Dict[str, Any], file_percentage: float) -> Optional[CoverageGap]:
        """Analyze a specific function's coverage gap."""
        try:
            func_percentage = func_data.get('percentage', 0)
            lines_covered = func_data.get('covered_lines', 0)
            lines_total = func_data.get('total_lines', 1)
            lines_uncovered = lines_total - lines_covered

            # Skip if already well-covered
            if func_percentage >= self.config.coverage_threshold:
                return None

            # Assess risk level
            risk_level = self._assess_risk_level(func_name, func_data, file_path)

            # Assess business impact
            business_impact = self._assess_business_impact(func_name, module_name)

            # Assess technical complexity
            technical_complexity = self._assess_technical_complexity(func_data)

            # Estimate test effort
            estimated_effort = self._estimate_test_effort(func_data, technical_complexity)

            # Find existing tests
            existing_tests = self._find_existing_tests(func_name, module_name)

            gap = CoverageGap(
                module_name=module_name,
                file_path=file_path,
                function_name=func_name,
                coverage_percentage=func_percentage,
                lines_uncovered=lines_uncovered,
                branches_uncovered=func_data.get('branches_uncovered', 0),
                risk_level=risk_level,
                business_impact=business_impact,
                technical_complexity=technical_complexity,
                estimated_test_effort=estimated_effort,
                existing_tests=existing_tests
            )

            return gap

        except Exception as e:
            logger.debug(f"Function gap analysis failed for {func_name}: {e}")
            return None

    def _assess_risk_level(self, func_name: str, func_data: Dict[str, Any], file_path: str) -> str:
        """Assess the risk level of a function based on various factors."""
        risk_score = 0

        # Function name analysis
        high_risk_names = ['auth', 'security', 'encrypt', 'decrypt', 'validate', 'authorize']
        if any(keyword in func_name.lower() for keyword in high_risk_names):
            risk_score += 3

        # File path analysis
        if any(keyword in file_path.lower() for keyword in ['security', 'auth', 'crypto', 'network']):
            risk_score += 2

        # Complexity analysis
        complexity = func_data.get('complexity', 1)
        if complexity > 10:
            risk_score += 2
        elif complexity > 5:
            risk_score += 1

        # External dependencies
        if func_data.get('external_calls', 0) > 0:
            risk_score += 1

        # Categorize risk
        if risk_score >= 5:
            return 'critical'
        elif risk_score >= 3:
            return 'high'
        elif risk_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _assess_business_impact(self, func_name: str, module_name: str) -> str:
        """Assess business impact of function coverage."""
        impact_score = 0

        # Core business functions
        core_functions = ['process', 'calculate', 'validate', 'execute', 'handle']
        if any(keyword in func_name.lower() for keyword in core_functions):
            impact_score += 2

        # Critical modules
        critical_modules = ['auth', 'payment', 'security', 'api', 'database']
        if any(keyword in module_name.lower() for keyword in critical_modules):
            impact_score += 2

        # Public interfaces
        if func_name.startswith(('get_', 'post_', 'put_', 'delete_')):
            impact_score += 1

        if impact_score >= 4:
            return 'high'
        elif impact_score >= 2:
            return 'medium'
        else:
            return 'low'

    def _assess_technical_complexity(self, func_data: Dict[str, Any]) -> str:
        """Assess technical complexity of function."""
        complexity = func_data.get('complexity', 1)
        branches = func_data.get('total_branches', 0)
        external_calls = func_data.get('external_calls', 0)

        complexity_score = complexity + branches//10 + external_calls

        if complexity_score > 15:
            return 'high'
        elif complexity_score > 8:
            return 'medium'
        else:
            return 'low'

    def _estimate_test_effort(self, func_data: Dict[str, Any], complexity: str) -> int:
        """Estimate effort required to test function (in hours)."""
        base_effort = 1

        # Complexity multiplier
        complexity_multiplier = {'low': 1, 'medium': 2, 'high': 4}
        base_effort *= complexity_multiplier[complexity]

        # Branches add effort
        branches = func_data.get('total_branches', 0)
        base_effort += branches // 20

        # External dependencies add effort
        external_calls = func_data.get('external_calls', 0)
        base_effort += external_calls // 5

        return max(1, min(8, base_effort))  # 1-8 hours

    def _find_existing_tests(self, func_name: str, module_name: str) -> List[str]:
        """Find existing tests for function."""
        # This would integrate with the test registry to find existing tests
        # For now, return empty list (would be populated by actual implementation)
        return []

    def _extract_module_name(self, file_path: str) -> str:
        """Extract module name from file path."""
        path = Path(file_path)
        if path.suffix == '.py':
            # Remove .py extension and convert path to module name
            parts = list(path.parts)
            if 'site-packages' in parts:
                # External package
                pkg_idx = parts.index('site-packages')
                return '.'.join(parts[pkg_idx + 1:])[:-3]  # Remove .py
            else:
                # Local module
                return '.'.join(path.parts)[:-3]  # Remove .py
        else:
            return path.stem

    def _build_dependency_graph(self, gaps: List[CoverageGap]):
        """Build dependency graph for strategic planning."""
        if not HAS_NETWORKX:
            # Simple fallback without networkx
            self.dependency_graph = None
            return

        self.dependency_graph.clear()

        # Add nodes
        for gap in gaps:
            self.dependency_graph.add_node(gap.function_name, gap=gap)

        # Add edges based on dependencies (simplified)
        # In a real implementation, this would analyze actual code dependencies
        for gap in gaps:
            # Find functions that this function calls
            for other_gap in gaps:
                if other_gap.function_name in gap.dependencies:
                    self.dependency_graph.add_edge(gap.function_name, other_gap.function_name)

    def _calculate_strategic_priorities(self, gaps: List[CoverageGap]):
        """Calculate strategic priority scores for all gaps."""
        for gap in gaps:
            # Risk score (0-1)
            risk_scores = {'critical': 1.0, 'high': 0.8, 'medium': 0.5, 'low': 0.2}
            risk_score = risk_scores[gap.risk_level]

            # Impact score (0-1)
            impact_scores = {'high': 1.0, 'medium': 0.6, 'low': 0.3}
            impact_score = impact_scores[gap.business_impact]

            # Complexity penalty (0-1, lower is better)
            complexity_scores = {'high': 0.3, 'medium': 0.6, 'low': 0.9}
            complexity_score = complexity_scores[gap.technical_complexity]

            # Effort normalization (lower effort = higher priority)
            effort_score = 1.0 / (1.0 + gap.estimated_test_effort)

            # Coverage gap bonus (lower coverage = higher priority)
            coverage_bonus = (100.0 - gap.coverage_percentage) / 100.0

            # Calculate weighted priority score
            priority_score = (
                self.strategy_params['risk_weight'] * risk_score +
                self.strategy_params['impact_weight'] * impact_score +
                self.strategy_params['complexity_weight'] * complexity_score +
                self.strategy_params['effort_weight'] * effort_score +
                coverage_bonus * 0.2  # Additional coverage bonus
            )

            gap.test_priority_score = min(1.0, priority_score)

    def predict_test_impact(self, test_vector: Any, gap: CoverageGap) -> CoveragePrediction:
        """
        Predict the coverage impact of a test vector on a coverage gap.

        Args:
            test_vector: Test vector to evaluate
            gap: Coverage gap being targeted

        Returns:
            Coverage impact prediction
        """
        try:
            confidence_score = 0.5  # Base confidence

            # Analyze test vector coverage targets
            coverage_targets = getattr(test_vector, 'coverage_targets', [])
            if gap.function_name in coverage_targets:
                confidence_score += 0.3

            # Analyze test type suitability
            test_type = getattr(test_vector, 'vector_type', 'unit')
            if self._is_test_type_suitable(test_type, gap):
                confidence_score += 0.2

            # Quality bonus
            quality_score = getattr(test_vector, 'quality_score', 0.5)
            confidence_score += quality_score * 0.1

            # Estimate coverage increase
            base_increase = gap.coverage_percentage * 0.3  # Conservative estimate
            expected_increase = base_increase * confidence_score

            # Risk assessment
            risk_assessment = self._assess_test_risk(test_vector, gap)

            # Dependencies check
            dependencies_satisfied = self._check_dependencies_satisfied(test_vector, gap)

            prediction = CoveragePrediction(
                test_vector_id=getattr(test_vector, 'name', 'unknown'),
                expected_coverage_increase=expected_increase,
                confidence_score=min(1.0, confidence_score),
                risk_assessment=risk_assessment,
                dependencies_satisfied=dependencies_satisfied,
                resource_requirements={
                    'estimated_time': gap.estimated_test_effort,
                    'complexity': gap.technical_complexity
                }
            )

            return prediction

        except Exception as e:
            logger.error(f"Test impact prediction failed: {e}")
            return CoveragePrediction(
                test_vector_id=getattr(test_vector, 'name', 'unknown'),
                expected_coverage_increase=0.0,
                confidence_score=0.0,
                risk_assessment='unknown',
                dependencies_satisfied=False
            )

    def _is_test_type_suitable(self, test_type: str, gap: CoverageGap) -> bool:
        """Check if test type is suitable for gap."""
        # Unit tests for unit gaps, integration tests for integration gaps, etc.
        if gap.risk_level == 'critical' and test_type == 'unit':
            return False  # Critical functions may need integration tests

        if gap.business_impact == 'high' and test_type == 'unit':
            return False  # High impact functions may need broader testing

        return True

    def _assess_test_risk(self, test_vector: Any, gap: CoverageGap) -> str:
        """Assess risk level of applying test to gap."""
        risk_level = 'low'

        # High-risk gaps need careful testing
        if gap.risk_level == 'critical':
            risk_level = 'high'

        # Complex tests are riskier
        if getattr(test_vector, 'complexity_level', 'medium') == 'high':
            risk_level = 'high'

        # New/unproven templates are riskier
        usage_count = getattr(test_vector, 'usage_count', 0)
        if usage_count < 5:
            risk_level = 'medium'

        return risk_level

    def _check_dependencies_satisfied(self, test_vector: Any, gap: CoverageGap) -> bool:
        """Check if test dependencies are satisfied."""
        # Simplified check - in real implementation would analyze actual dependencies
        test_deps = getattr(test_vector, 'dependencies', [])
        gap_deps = gap.dependencies

        return all(dep in test_deps or dep in gap_deps for dep in gap_deps)

    def create_optimization_strategy(self, gaps: List[CoverageGap],
                                   available_resources: Dict[str, Any]) -> OptimizationStrategy:
        """
        Create an optimal test generation strategy based on gaps and resources.

        Args:
            gaps: Analyzed coverage gaps
            available_resources: Available testing resources

        Returns:
            Strategic optimization plan
        """
        try:
            # Filter high-priority gaps
            high_priority_gaps = [g for g in gaps if g.test_priority_score >= self.strategy_params['min_priority_score']]

            # Limit scope for manageability
            target_gaps = high_priority_gaps[:self.strategy_params['max_strategy_modules']]
            target_modules = list(set(g.module_name for g in target_gaps))

            # Calculate priority order
            priority_order = sorted(target_modules,
                                  key=lambda m: statistics.mean(g.test_priority_score
                                                               for g in target_gaps if g.module_name == m),
                                  reverse=True)

            # Estimate completion time
            total_effort = sum(g.estimated_test_effort for g in target_gaps)
            estimated_completion_time = total_effort  # 1:1 effort to time ratio

            # Estimate coverage gain
            avg_gap_coverage = statistics.mean(g.coverage_percentage for g in target_gaps)
            expected_coverage_gain = (100.0 - avg_gap_coverage) * 0.7  # Conservative estimate

            # Create risk mitigation plan
            risk_mitigation = self._create_risk_mitigation_plan(target_gaps)

            # Resource requirements
            resource_requirements = self._calculate_resource_requirements(target_gaps, available_resources)

            # Success probability
            success_probability = self._calculate_success_probability(target_gaps, available_resources)

            strategy = OptimizationStrategy(
                strategy_id=f"strategy_{int(os.times().elapsed)}",
                target_modules=target_modules,
                priority_order=priority_order,
                estimated_completion_time=estimated_completion_time,
                expected_coverage_gain=expected_coverage_gain,
                risk_mitigation_plan=risk_mitigation,
                resource_requirements=resource_requirements,
                success_probability=success_probability
            )

            logger.info(f"Created optimization strategy: {len(target_modules)} modules, "
                       f"{estimated_completion_time}h effort, {expected_coverage_gain:.1f}% coverage gain")

            return strategy

        except Exception as e:
            logger.error(f"Strategy creation failed: {e}")
            # Return minimal fallback strategy
            return OptimizationStrategy(
                strategy_id="fallback_strategy",
                target_modules=[],
                priority_order=[],
                estimated_completion_time=0,
                expected_coverage_gain=0.0,
                risk_mitigation_plan={},
                resource_requirements={},
                success_probability=0.0
            )

    def _create_risk_mitigation_plan(self, gaps: List[CoverageGap]) -> Dict[str, Any]:
        """Create risk mitigation plan for strategy."""
        mitigation_plan = {
            'critical_gaps': [],
            'fallback_strategies': {},
            'monitoring_points': [],
            'rollback_triggers': []
        }

        # Identify critical gaps
        critical_gaps = [g for g in gaps if g.risk_level == 'critical']
        mitigation_plan['critical_gaps'] = [g.function_name for g in critical_gaps]

        # Create fallback strategies
        for gap in critical_gaps:
            mitigation_plan['fallback_strategies'][gap.function_name] = {
                'manual_testing_required': True,
                'simplified_test_acceptable': gap.technical_complexity != 'high',
                'peer_review_mandatory': True
            }

        # Define monitoring points
        total_effort = sum(g.estimated_test_effort for g in gaps)
        monitoring_intervals = [total_effort * i // 4 for i in range(1, 4)]
        mitigation_plan['monitoring_points'] = monitoring_intervals

        # Rollback triggers
        mitigation_plan['rollback_triggers'] = [
            'coverage_below_70_percent',
            'critical_test_failures_above_20_percent',
            'resource_exhaustion_detected'
        ]

        return mitigation_plan

    def _calculate_resource_requirements(self, gaps: List[CoverageGap],
                                       available_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate resource requirements for strategy."""
        total_effort = sum(g.estimated_test_effort for g in gaps)

        # Estimate parallel workers needed
        avg_effort_per_gap = statistics.mean(g.estimated_test_effort for g in gaps)
        optimal_workers = min(8, max(1, len(gaps) // 5))  # 1 worker per 5 gaps, max 8

        return {
            'developer_hours': total_effort,
            'parallel_workers': optimal_workers,
            'estimated_duration_days': total_effort // 8,  # Assuming 8 hours/day
            'peak_memory_mb': 500 * optimal_workers,  # Estimate
            'storage_mb': 100 * len(gaps)  # Test artifacts
        }

    def _calculate_success_probability(self, gaps: List[CoverageGap],
                                     available_resources: Dict[str, Any]) -> float:
        """Calculate probability of strategy success."""
        base_probability = 0.8  # Base success rate

        # Adjust for gap characteristics
        avg_complexity = statistics.mean({'low': 1, 'medium': 2, 'high': 3}[g.technical_complexity] for g in gaps)
        complexity_penalty = (avg_complexity - 1) * 0.1
        base_probability -= complexity_penalty

        # Adjust for resource availability
        required_workers = len(gaps) // 5
        available_workers = available_resources.get('parallel_workers', 1)
        resource_penalty = max(0, (required_workers - available_workers) * 0.1)
        base_probability -= resource_penalty

        # Adjust for risk level
        high_risk_gaps = sum(1 for g in gaps if g.risk_level in ['critical', 'high'])
        risk_penalty = high_risk_gaps / len(gaps) * 0.2
        base_probability -= risk_penalty

        return max(0.1, min(1.0, base_probability))

    def get_coverage_insights(self) -> Dict[str, Any]:
        """Get strategic insights about coverage optimization."""
        insights = {
            'risk_distribution': {},
            'impact_distribution': {},
            'complexity_distribution': {},
            'effort_distribution': {},
            'strategic_recommendations': []
        }

        if not self.gap_analysis_cache:
            return insights

        gaps = list(self.gap_analysis_cache.values())

        # Analyze distributions
        insights['risk_distribution'] = dict(Counter(g.risk_level for g in gaps))
        insights['impact_distribution'] = dict(Counter(g.business_impact for g in gaps))
        insights['complexity_distribution'] = dict(Counter(g.technical_complexity for g in gaps))

        # Effort analysis
        efforts = [g.estimated_test_effort for g in gaps]
        insights['effort_distribution'] = {
            'total_hours': sum(efforts),
            'average_hours': statistics.mean(efforts) if efforts else 0,
            'max_hours': max(efforts) if efforts else 0
        }

        # Strategic recommendations
        insights['strategic_recommendations'] = self._generate_strategic_recommendations(gaps)

        return insights

    def _generate_strategic_recommendations(self, gaps: List[CoverageGap]) -> List[str]:
        """Generate strategic recommendations for coverage optimization."""
        recommendations = []

        if not gaps:
            return recommendations

        # Risk-based recommendations
        critical_gaps = [g for g in gaps if g.risk_level == 'critical']
        if critical_gaps:
            recommendations.append(f"Prioritize {len(critical_gaps)} critical risk gaps immediately")

        # Effort optimization
        high_effort_gaps = [g for g in gaps if g.estimated_test_effort > 4]
        if high_effort_gaps:
            recommendations.append(f"Consider breaking down {len(high_effort_gaps)} high-effort gaps into smaller tests")

        # Complexity analysis
        high_complexity_gaps = [g for g in gaps if g.technical_complexity == 'high']
        if high_complexity_gaps:
            recommendations.append(f"Allocate senior developers to {len(high_complexity_gaps)} high-complexity gaps")

        # Coverage target analysis
        low_coverage_modules = {}
        for gap in gaps:
            if gap.module_name not in low_coverage_modules:
                low_coverage_modules[gap.module_name] = []
            low_coverage_modules[gap.module_name].append(gap.coverage_percentage)

        modules_needing_attention = [
            module for module, coverages in low_coverage_modules.items()
            if statistics.mean(coverages) < 50.0
        ]
        if modules_needing_attention:
            recommendations.append(f"Focus on {len(modules_needing_attention)} modules with coverage below 50%")

        return recommendations

    def export_strategy_report(self, strategy: OptimizationStrategy, output_path: Path):
        """Export detailed strategy report."""
        report = {
            'strategy_id': strategy.strategy_id,
            'overview': {
                'target_modules': strategy.target_modules,
                'priority_order': strategy.priority_order,
                'estimated_completion_time': strategy.estimated_completion_time,
                'expected_coverage_gain': strategy.expected_coverage_gain,
                'success_probability': strategy.success_probability
            },
            'risk_mitigation': strategy.risk_mitigation_plan,
            'resource_requirements': strategy.resource_requirements,
            'execution_plan': {
                'phase_1_focus': strategy.priority_order[:len(strategy.priority_order)//3],
                'phase_2_focus': strategy.priority_order[len(strategy.priority_order)//3:2*len(strategy.priority_order)//3],
                'phase_3_focus': strategy.priority_order[2*len(strategy.priority_order)//3:]
            },
            'success_metrics': {
                'target_coverage_threshold': self.strategy_params['coverage_target_threshold'],
                'minimum_priority_score': self.strategy_params['min_priority_score'],
                'risk_mitigation_coverage': len(strategy.risk_mitigation_plan.get('critical_gaps', []))
            }
        }

        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"Strategy report exported to {output_path}")


# Integration functions
def analyze_coverage_strategically(coverage_data: Dict[str, Any], config) -> Dict[str, Any]:
    """Perform strategic coverage analysis."""
    engine = CoverageAnalysisEngine(config)
    gaps = engine.analyze_coverage_gaps(coverage_data)
    insights = engine.get_coverage_insights()

    return {
        'gaps': [gap.__dict__ for gap in gaps],
        'insights': insights,
        'total_gaps': len(gaps),
        'high_priority_gaps': len([g for g in gaps if g.test_priority_score >= 0.8])
    }


def create_coverage_optimization_strategy(gaps_data: List[Dict[str, Any]], config) -> OptimizationStrategy:
    """Create coverage optimization strategy."""
    engine = CoverageAnalysisEngine(config)

    # Convert dicts back to CoverageGap objects
    gaps = [CoverageGap(**gap_dict) for gap_dict in gaps_data]

    # Mock available resources (would be detected in real implementation)
    available_resources = {
        'parallel_workers': 4,
        'developer_hours_available': 160,  # 4 weeks * 5 days * 8 hours
        'memory_gb': 16,
        'storage_gb': 100
    }

    return engine.create_optimization_strategy(gaps, available_resources)
