"""
Template Evolution & Learning System for VectorReVamp

Self-improving template system that learns from generation success/failure,
evolves templates through usage patterns, and adapts to new codebases.
Inspired by evolutionary algorithms and machine learning adaptation.
"""

import os
import json
import time
import random
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import hashlib
import statistics

# Forward references to avoid circular imports
TestTemplate = None  # Will be resolved at runtime when needed

logger = logging.getLogger(__name__)


@dataclass
class GenerationFeedback:
    """Feedback from test generation attempts."""
    template_id: str
    module_name: str
    success: bool
    quality_score: float
    execution_time: float
    error_message: str = ""
    test_vectors_generated: int = 0
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemplateEvolutionMetrics:
    """Metrics tracking template evolution."""
    template_id: str
    total_usage: int = 0
    success_count: int = 0
    average_quality: float = 0.0
    average_execution_time: float = 0.0
    failure_rate: float = 0.0
    adaptation_score: float = 0.0
    last_used: float = 0.0
    evolution_attempts: int = 0
    successful_mutations: int = 0


@dataclass
class EvolutionStrategy:
    """Strategy for template evolution."""
    strategy_type: str  # 'mutation', 'crossover', 'specialization', 'generalization'
    template_id: str
    changes: Dict[str, Any]
    expected_improvement: float
    risk_level: str  # 'low', 'medium', 'high'
    prerequisites: List[str] = field(default_factory=list)


class TemplateEvolutionEngine:
    """
    Self-improving template evolution system.

    Learns from generation patterns and evolves templates through:
    - Success-based reinforcement learning
    - Genetic algorithm-inspired mutation and crossover
    - Domain specialization and generalization
    - Pattern discovery from new codebases
    """

    def __init__(self, config):
        self.config = config

        # Evolution state
        self.feedback_history: List[GenerationFeedback] = []
        self.evolution_metrics: Dict[str, TemplateEvolutionMetrics] = {}
        self.template_lineage: Dict[str, List[str]] = {}  # template -> [parent_templates]

        # Evolution parameters
        self.evolution_params = {
            'min_feedback_samples': 10,  # Minimum samples before evolution
            'evolution_interval': 3600,  # 1 hour between evolution attempts
            'mutation_rate': 0.1,        # Probability of mutation
            'crossover_rate': 0.05,      # Probability of crossover
            'quality_threshold': 0.7,    # Minimum quality for successful evolution
            'risk_tolerance': 0.8,       # Maximum acceptable risk
        }

        # Learning state
        self.last_evolution_time = 0.0
        self.success_patterns: Dict[str, Dict[str, Any]] = {}
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}

        logger.info("TemplateEvolutionEngine initialized")

    def record_feedback(self, feedback: GenerationFeedback):
        """Record generation feedback for learning."""
        self.feedback_history.append(feedback)

        # Update metrics
        if feedback.template_id not in self.evolution_metrics:
            self.evolution_metrics[feedback.template_id] = TemplateEvolutionMetrics(
                template_id=feedback.template_id
            )

        metrics = self.evolution_metrics[feedback.template_id]
        metrics.total_usage += 1
        metrics.last_used = feedback.timestamp

        if feedback.success:
            metrics.success_count += 1

        # Update rolling averages
        self._update_metrics_averages(metrics, feedback)

        # Learn from patterns
        self._learn_from_feedback(feedback)

        logger.debug(f"Recorded feedback for template {feedback.template_id}: "
                    f"success={feedback.success}, quality={feedback.quality_score:.2f}")

    def _update_metrics_averages(self, metrics: TemplateEvolutionMetrics, feedback: GenerationFeedback):
        """Update rolling averages for metrics."""
        # Simple exponential moving average for quality
        alpha = 0.1  # Learning rate
        metrics.average_quality = (1 - alpha) * metrics.average_quality + alpha * feedback.quality_score

        # Execution time average
        if metrics.average_execution_time == 0:
            metrics.average_execution_time = feedback.execution_time
        else:
            metrics.average_execution_time = (1 - alpha) * metrics.average_execution_time + alpha * feedback.execution_time

        # Failure rate
        metrics.failure_rate = 1.0 - (metrics.success_count / metrics.total_usage)

    def _learn_from_feedback(self, feedback: GenerationFeedback):
        """Learn patterns from generation feedback."""
        pattern_key = self._generate_pattern_key(feedback)

        if feedback.success:
            if pattern_key not in self.success_patterns:
                self.success_patterns[pattern_key] = {
                    'count': 0,
                    'avg_quality': 0.0,
                    'contexts': []
                }

            pattern = self.success_patterns[pattern_key]
            pattern['count'] += 1
            pattern['avg_quality'] = (pattern['avg_quality'] * (pattern['count'] - 1) + feedback.quality_score) / pattern['count']
            pattern['contexts'].append(feedback.context)

            # Keep only recent contexts
            if len(pattern['contexts']) > 10:
                pattern['contexts'] = pattern['contexts'][-10:]

        else:
            if pattern_key not in self.failure_patterns:
                self.failure_patterns[pattern_key] = {
                    'count': 0,
                    'errors': []
                }

            pattern = self.failure_patterns[pattern_key]
            pattern['count'] += 1
            if feedback.error_message:
                pattern['errors'].append(feedback.error_message)

    def _generate_pattern_key(self, feedback: GenerationFeedback) -> str:
        """Generate a key for pattern matching."""
        # Create pattern key from context features
        context = feedback.context
        key_parts = [
            feedback.template_id,
            context.get('language', 'unknown'),
            context.get('framework', 'unknown'),
            str(context.get('complexity', 0)),
            str(len(context.get('dependencies', []))),
        ]
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()[:8]

    def should_evolve(self, template_id: str) -> bool:
        """Determine if a template should be evolved."""
        if template_id not in self.evolution_metrics:
            return False

        metrics = self.evolution_metrics[template_id]

        # Check minimum usage threshold
        if metrics.total_usage < self.evolution_params['min_feedback_samples']:
            return False

        # Check time since last evolution
        current_time = time.time()
        if current_time - self.last_evolution_time < self.evolution_params['evolution_interval']:
            return False

        # Check if evolution would be beneficial
        if metrics.average_quality < 0.8 and metrics.failure_rate > 0.2:
            return True

        # Check for recent decline in performance
        recent_feedbacks = [f for f in self.feedback_history[-20:]
                          if f.template_id == template_id and f.timestamp > current_time - 3600]
        if len(recent_feedbacks) >= 5:
            recent_avg_quality = sum(f.quality_score for f in recent_feedbacks) / len(recent_feedbacks)
            if recent_avg_quality < metrics.average_quality * 0.9:  # 10% decline
                return True

        return False

    def evolve_template(self, template) -> Optional[Any]:
        """
        Evolve a template based on learning and feedback.

        Returns a new evolved template or None if evolution fails.
        """
        try:
            logger.info(f"Attempting to evolve template {template.template_id}")

            # Generate evolution strategies
            strategies = self._generate_evolution_strategies(template)

            if not strategies:
                logger.debug(f"No viable evolution strategies for template {template.template_id}")
                return None

            # Select best strategy
            best_strategy = self._select_best_strategy(strategies, template)

            if not best_strategy:
                return None

            # Apply evolution
            evolved_template = self._apply_evolution_strategy(template, best_strategy)

            if evolved_template:
                # Validate evolution
                if self._validate_evolution(evolved_template, template):
                    # Record lineage
                    if evolved_template.template_id not in self.template_lineage:
                        self.template_lineage[evolved_template.template_id] = []
                    self.template_lineage[evolved_template.template_id].append(template.template_id)

                    # Update metrics
                    metrics = self.evolution_metrics[template.template_id]
                    metrics.evolution_attempts += 1

                    self.last_evolution_time = time.time()

                    logger.info(f"Successfully evolved template {template.template_id} -> {evolved_template.template_id}")
                    return evolved_template
                else:
                    logger.debug(f"Evolution validation failed for template {template.template_id}")
                    return None
            else:
                logger.debug(f"Evolution application failed for template {template.template_id}")
                return None

        except Exception as e:
            logger.error(f"Template evolution failed for {template.template_id}: {e}")
            return None

    def _generate_evolution_strategies(self, template) -> List[EvolutionStrategy]:
        """Generate potential evolution strategies for a template."""
        strategies = []

        # Strategy 1: Mutation - modify existing patterns
        if random.random() < self.evolution_params['mutation_rate']:
            strategies.append(EvolutionStrategy(
                strategy_type='mutation',
                template_id=template.template_id,
                changes=self._generate_mutation_changes(template),
                expected_improvement=0.1,
                risk_level='low'
            ))

        # Strategy 2: Crossover - combine with successful patterns
        if random.random() < self.evolution_params['crossover_rate']:
            crossover_changes = self._generate_crossover_changes(template)
            if crossover_changes:
                strategies.append(EvolutionStrategy(
                    strategy_type='crossover',
                    template_id=template.template_id,
                    changes=crossover_changes,
                    expected_improvement=0.15,
                    risk_level='medium'
                ))

        # Strategy 3: Specialization - adapt to successful contexts
        success_patterns = self._find_success_patterns(template)
        if success_patterns:
            strategies.append(EvolutionStrategy(
                strategy_type='specialization',
                template_id=template.template_id,
                changes=self._generate_specialization_changes(template, success_patterns),
                expected_improvement=0.2,
                risk_level='low'
            ))

        # Strategy 4: Generalization - broaden applicability
        if template.usage_count > 50:  # Only generalize mature templates
            strategies.append(EvolutionStrategy(
                strategy_type='generalization',
                template_id=template.template_id,
                changes=self._generate_generalization_changes(template),
                expected_improvement=0.05,
                risk_level='high'
            ))

        return strategies

    def _generate_mutation_changes(self, template) -> Dict[str, Any]:
        """Generate mutation changes for template evolution."""
        changes = {}

        # Mutate patterns with small random changes
        if template.setup_patterns and random.random() < 0.3:
            # Add or modify setup pattern
            mutation_idx = random.randint(0, len(template.setup_patterns) - 1)
            original = template.setup_patterns[mutation_idx]
            # Simple mutation: add optional elements
            changes['setup_patterns'] = template.setup_patterns.copy()
            changes['setup_patterns'][mutation_idx] = original + " # Enhanced setup"

        if template.assertion_patterns and random.random() < 0.4:
            # Enhance assertion pattern
            assertion_idx = random.randint(0, len(template.assertion_patterns) - 1)
            original = template.assertion_patterns[assertion_idx]
            # Add more comprehensive assertion
            changes['assertion_patterns'] = template.assertion_patterns.copy()
            if 'assert' in original and 'is not None' in original:
                changes['assertion_patterns'][assertion_idx] = original + " and len(result) > 0"

        # Adjust quality score expectation
        if random.random() < 0.2:
            changes['quality_score'] = min(1.0, template.quality_score + 0.05)

        return changes

    def _generate_crossover_changes(self, template) -> Optional[Dict[str, Any]]:
        """Generate crossover changes by combining with successful patterns."""
        # Find successful patterns from other templates
        successful_patterns = []
        for pattern_key, pattern_data in self.success_patterns.items():
            if pattern_data['count'] >= 3 and pattern_data['avg_quality'] > 0.8:
                successful_patterns.append(pattern_data)

        if not successful_patterns:
            return None

        # Select random successful pattern
        selected_pattern = random.choice(successful_patterns)

        changes = {}

        # Cross over assertion patterns
        if selected_pattern.get('contexts'):
            context = random.choice(selected_pattern['contexts'])
            if context.get('successful_assertions'):
                changes['assertion_patterns'] = template.assertion_patterns + context['successful_assertions'][:2]

        return changes

    def _generate_specialization_changes(self, template,
                                       success_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate specialization changes based on successful usage patterns."""
        changes = {}

        # Analyze successful contexts
        languages = Counter()
        frameworks = Counter()
        complexities = []

        for pattern in success_patterns:
            for context in pattern.get('contexts', []):
                languages[context.get('language', 'unknown')] += 1
                frameworks[context.get('framework', 'unknown')] += 1
                complexities.append(context.get('complexity', 1))

        # Specialize for most common language/framework
        most_common_lang = languages.most_common(1)[0][0] if languages else None
        most_common_framework = frameworks.most_common(1)[0][0] if frameworks else None

        if most_common_lang and most_common_lang != template.language:
            changes['language'] = most_common_lang

        if most_common_framework and most_common_framework != template.framework:
            changes['framework'] = most_common_framework

        # Adjust complexity level
        if complexities:
            avg_complexity = sum(complexities) / len(complexities)
            if avg_complexity > template.complexity_level:
                changes['complexity_level'] = 'complex'
            elif avg_complexity < template.complexity_level:
                changes['complexity_level'] = 'simple'

        return changes

    def _generate_generalization_changes(self, template) -> Dict[str, Any]:
        """Generate generalization changes to broaden template applicability."""
        changes = {}

        # Broaden language/framework support (risky)
        if random.random() < 0.1:  # Low probability
            if template.language == 'python':
                changes['language'] = 'universal'  # Allow broader usage
            elif template.framework == 'pytest':
                changes['framework'] = 'universal'

        # Relax quality requirements slightly
        if template.quality_score > 0.8:
            changes['quality_score'] = template.quality_score - 0.05  # More permissive

        # Add fallback patterns
        if template.assertion_patterns:
            fallback_assertions = ["assert result is not None  # Fallback check"]
            changes['assertion_patterns'] = template.assertion_patterns + fallback_assertions

        return changes

    def _find_success_patterns(self, template) -> List[Dict[str, Any]]:
        """Find successful usage patterns for a template."""
        template_success_patterns = []

        for pattern_key, pattern_data in self.success_patterns.items():
            if pattern_data['count'] >= 3:  # At least 3 successful uses
                # Check if pattern is related to this template
                for context in pattern_data.get('contexts', []):
                    if (context.get('language') == template.language or
                        context.get('framework') == template.framework):
                        template_success_patterns.append(pattern_data)
                        break

        return template_success_patterns

    def _select_best_strategy(self, strategies: List[EvolutionStrategy],
                            template) -> Optional[EvolutionStrategy]:
        """Select the best evolution strategy."""
        if not strategies:
            return None

        # Score strategies based on expected improvement and risk
        scored_strategies = []
        for strategy in strategies:
            # Calculate composite score
            improvement_score = strategy.expected_improvement

            # Risk adjustment
            risk_penalty = {'low': 0.0, 'medium': 0.1, 'high': 0.2}
            risk_adjusted_score = improvement_score - risk_penalty[strategy.risk_level]

            # Current template performance bonus
            metrics = self.evolution_metrics.get(template.template_id)
            if metrics and metrics.average_quality < 0.7:
                risk_adjusted_score += 0.1  # More aggressive evolution for poor performers

            scored_strategies.append((risk_adjusted_score, strategy))

        # Select highest scoring strategy
        scored_strategies.sort(reverse=True)
        best_score, best_strategy = scored_strategies[0]

        # Only proceed if score is positive
        if best_score > 0:
            return best_strategy

        return None

    def _apply_evolution_strategy(self, template,
                                strategy: EvolutionStrategy) -> Optional[Any]:
        """Apply an evolution strategy to create a new template."""
        try:
            # Create evolved template
            evolved_template = TestTemplate(
                template_id="",  # Will be set below
                test_type=template.test_type,
                language=template.language,
                framework=template.framework,
                structure=template.structure.copy(),
                setup_patterns=template.setup_patterns.copy(),
                test_patterns=template.test_patterns.copy(),
                assertion_patterns=template.assertion_patterns.copy(),
                cleanup_patterns=template.cleanup_patterns.copy(),
                extracted_from=template.extracted_from.copy(),
                quality_score=template.quality_score,
                usage_count=0,  # New template starts fresh
                success_rate=0.0,
                complexity_level=template.complexity_level,
                coverage_targets=template.coverage_targets.copy(),
                parameter_patterns=template.parameter_patterns.copy(),
                validation_rules=template.validation_rules.copy()
            )

            # Apply changes
            for change_key, change_value in strategy.changes.items():
                if hasattr(evolved_template, change_key):
                    setattr(evolved_template, change_key, change_value)

            # Generate new template ID
            evolved_template.template_id = self._generate_evolved_template_id(template, strategy)

            return evolved_template

        except Exception as e:
            logger.error(f"Failed to apply evolution strategy: {e}")
            return None

    def _generate_evolved_template_id(self, original_template,
                                    strategy: EvolutionStrategy) -> str:
        """Generate ID for evolved template."""
        base_id = original_template.template_id
        evolution_marker = f"{strategy.strategy_type}_{int(time.time())}"
        return f"{base_id}_evolved_{evolution_marker}"

    def _validate_evolution(self, evolved_template,
                          original_template) -> bool:
        """Validate that evolution produced a viable template."""
        try:
            # Basic validation checks
            if not evolved_template.template_id:
                return False

            if not evolved_template.assertion_patterns and not evolved_template.test_patterns:
                return False

            # Ensure quality score is reasonable
            if evolved_template.quality_score < 0.3:
                return False

            # Check that evolved template is different from original
            changes_detected = (
                evolved_template.setup_patterns != original_template.setup_patterns or
                evolved_template.test_patterns != original_template.test_patterns or
                evolved_template.assertion_patterns != original_template.assertion_patterns or
                evolved_template.quality_score != original_template.quality_score
            )

            if not changes_detected:
                return False

            return True

        except Exception as e:
            logger.error(f"Evolution validation failed: {e}")
            return False

    def get_evolution_recommendations(self) -> List[Dict[str, Any]]:
        """Get recommendations for template evolution."""
        recommendations = []

        for template_id, metrics in self.evolution_metrics.items():
            if self.should_evolve(template_id):
                recommendations.append({
                    'template_id': template_id,
                    'reason': 'performance_improvement_needed',
                    'current_quality': metrics.average_quality,
                    'failure_rate': metrics.failure_rate,
                    'usage_count': metrics.total_usage,
                    'recommended_action': 'evolve_template'
                })

        # Sort by priority (highest failure rate first)
        recommendations.sort(key=lambda x: (x['failure_rate'], -x['current_quality']), reverse=True)

        return recommendations[:10]  # Top 10 recommendations

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evolution statistics."""
        stats = {
            'total_templates_evolved': len(self.template_lineage),
            'total_evolution_attempts': sum(m.evolution_attempts for m in self.evolution_metrics.values()),
            'successful_evolutions': sum(m.successful_mutations for m in self.evolution_metrics.values()),
            'evolution_success_rate': 0.0,
            'average_quality_improvement': 0.0,
            'most_successful_strategies': {},
            'template_lineage_depth': {},
        }

        # Calculate success rate
        if stats['total_evolution_attempts'] > 0:
            stats['evolution_success_rate'] = stats['successful_evolutions'] / stats['total_evolution_attempts']

        # Calculate lineage depth
        for evolved_id, parents in self.template_lineage.items():
            depth = len(parents)  # Simplified depth calculation
            stats['template_lineage_depth'][evolved_id] = depth

        return stats

    def save_evolution_state(self, output_path: Path):
        """Save evolution state for persistence."""
        state = {
            'feedback_history': [f.__dict__ for f in self.feedback_history[-1000:]],  # Last 1000 feedbacks
            'evolution_metrics': {tid: m.__dict__ for tid, m in self.evolution_metrics.items()},
            'template_lineage': self.template_lineage,
            'success_patterns': self.success_patterns,
            'failure_patterns': self.failure_patterns,
            'last_evolution_time': self.last_evolution_time,
            'evolution_params': self.evolution_params,
        }

        with open(output_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

        logger.info(f"Saved evolution state to {output_path}")

    def load_evolution_state(self, input_path: Path):
        """Load evolution state from file."""
        try:
            with open(input_path, 'r') as f:
                state = json.load(f)

            # Restore state
            self.template_lineage = state.get('template_lineage', {})
            self.success_patterns = state.get('success_patterns', {})
            self.failure_patterns = state.get('failure_patterns', {})
            self.last_evolution_time = state.get('last_evolution_time', 0.0)
            self.evolution_params = state.get('evolution_params', self.evolution_params)

            # Restore metrics
            for tid, m_dict in state.get('evolution_metrics', {}).items():
                self.evolution_metrics[tid] = TemplateEvolutionMetrics(**m_dict)

            # Restore feedback history (limited)
            for f_dict in state.get('feedback_history', []):
                feedback = GenerationFeedback(**f_dict)
                self.feedback_history.append(feedback)

            logger.info(f"Loaded evolution state from {input_path}")

        except Exception as e:
            logger.error(f"Failed to load evolution state: {e}")


# Integration functions
def create_evolution_feedback(template_id: str, module_name: str, success: bool,
                            quality_score: float, execution_time: float,
                            error_message: str = "", test_vectors_generated: int = 0,
                            context: Dict[str, Any] = None) -> GenerationFeedback:
    """Create a generation feedback object."""
    return GenerationFeedback(
        template_id=template_id,
        module_name=module_name,
        success=success,
        quality_score=quality_score,
        execution_time=execution_time,
        error_message=error_message,
        test_vectors_generated=test_vectors_generated,
        context=context or {}
    )


def evolve_templates_if_needed(evolution_engine: TemplateEvolutionEngine,
                             templates: Dict[str, Any]) -> Dict[str, Any]:
    """Evolve templates that need improvement."""
    evolved_templates = {}

    recommendations = evolution_engine.get_evolution_recommendations()

    for rec in recommendations:
        template_id = rec['template_id']
        if template_id in templates:
            original_template = templates[template_id]
            evolved_template = evolution_engine.evolve_template(original_template)

            if evolved_template:
                evolved_templates[evolved_template.template_id] = evolved_template
                logger.info(f"Evolved template {template_id} -> {evolved_template.template_id}")

    return evolved_templates
