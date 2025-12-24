"""
Main Test Harness Runner

Orchestrates test execution, coverage analysis, and test generation.
Framework-agnostic implementation.
"""

import subprocess
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .test_vector import TestVectorRegistry, TestVector
from .coverage_analyzer import CoverageAnalyzer
from .code_embedder import CodeEmbedder
from .llm_generator import LLMTestGenerator
from .template_engine import TemplateEngine, TestTemplate, CodeContext
from .parallel_engine import ParallelGenerationEngine, GenerationTask
from .coverage_optimizer import CoverageAnalysisEngine, analyze_coverage_strategically
from .plugin_system import PluginRegistry, PluginLoader, PluginOrchestrator, PythonLanguagePlugin
from .config import HarnessConfig

# Configure logging
logger = logging.getLogger(__name__)


class TestHarnessRunner:
    """Main test harness runner"""
    
    def __init__(self, config: HarnessConfig):
        """
        Initialize test harness runner
        
        Args:
            config: HarnessConfig instance
        """
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry = TestVectorRegistry()
        self.coverage_analyzer = CoverageAnalyzer(config)
        self.code_embedder = CodeEmbedder(config)
        self.llm_generator = LLMTestGenerator(config, self.coverage_analyzer, self.code_embedder)
        self.template_engine = TemplateEngine(config)
        self.parallel_engine = ParallelGenerationEngine(config)
        self.coverage_optimizer = CoverageAnalysisEngine(config)

        # Plugin system
        self.plugin_registry = PluginRegistry()
        self.plugin_loader = PluginLoader(self.plugin_registry)
        self.plugin_orchestrator = PluginOrchestrator(self.plugin_registry)

        # Load built-in plugins
        self._load_builtin_plugins()

    def _load_builtin_plugins(self):
        """Load built-in plugins."""
        try:
            # Load Python language plugin
            python_plugin = PythonLanguagePlugin()
            if self.plugin_registry.register_plugin(python_plugin):
                python_plugin.initialize({'language': 'python'})
                logger.info("Python language plugin loaded")

            # Load other built-in plugins
            try:
                # Load framework plugins
                framework_plugins = [
                    ('pytest', 'pytest'),
                    ('unittest', 'unittest'),
                    ('cmocka', 'cmocka')
                ]

                for plugin_name, framework in framework_plugins:
                    try:
                        plugin = self.plugin_registry.create_plugin(
                            'framework',
                            plugin_name,
                            {'framework': framework}
                        )
                        if plugin:
                            plugin.initialize({'framework': framework})
                            logger.info(f"Framework plugin loaded: {plugin_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load framework plugin {plugin_name}: {e}")

                # Load domain plugins
                domain_plugins = [
                    ('security', 'security'),
                    ('performance', 'performance'),
                    ('database', 'database'),
                    ('web_api', 'web_api')
                ]

                for plugin_name, domain in domain_plugins:
                    try:
                        plugin = self.plugin_registry.create_plugin(
                            'domain',
                            plugin_name,
                            {'domain': domain}
                        )
                        if plugin:
                            plugin.initialize({'domain': domain})
                            logger.info(f"Domain plugin loaded: {plugin_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load domain plugin {plugin_name}: {e}")

                # Load additional language plugins
                language_plugins = [
                    ('rust', 'rust'),
                    ('c', 'c'),
                    ('cpp', 'cpp')
                ]

                for plugin_name, language in language_plugins:
                    try:
                        plugin = self.plugin_registry.create_plugin(
                            'language',
                            plugin_name,
                            {'language': language}
                        )
                        if plugin:
                            plugin.initialize({'language': language})
                            logger.info(f"Language plugin loaded: {plugin_name}")
                    except Exception as e:
                        logger.warning(f"Failed to load language plugin {plugin_name}: {e}")

            except Exception as e:
                logger.error(f"Failed to load additional plugins: {e}")

        except Exception as e:
            logger.error(f"Failed to load built-in plugins: {e}")
        
        self.results: Dict[str, Any] = {
            'run_timestamp': datetime.now().isoformat(),
            'test_results': {},
            'coverage_report': {},
            'generated_vectors': [],
        }
    
    def initialize(self):
        """Initialize vector database with codebase and tests"""
        logger.info("=" * 70)
        logger.info("Unified Test Harness - Initialization")
        logger.info("=" * 70)
        
        try:
            if self.config.use_vector_db:
                logger.info("\n[1/3] Embedding codebase...")
                self.code_embedder.embed_codebase()

                logger.info("\n[2/3] Embedding test templates...")
                self.code_embedder.embed_existing_tests()

                logger.info("\n[3/3] Extracting test templates...")
                self.template_engine.extract_templates(self.config.source_root)
                template_stats = self.template_engine.get_template_statistics()
                logger.info(f"Extracted {template_stats['total_templates']} templates "
                          f"(avg quality: {template_stats['average_quality_score']:.2f})")
            else:
                logger.warning("\nVector database disabled, skipping embedding and template extraction")

            logger.info("\n[+] Initialization complete!")
        except Exception as e:
            logger.error(f"Initialization failed: {e}", exc_info=True)
            raise
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis"""
        logger.info("Running coverage analysis...")
        
        try:
            coverage_report = self.coverage_analyzer.run_coverage_analysis()
            self.results['coverage_report'] = coverage_report
            
            if coverage_report:
                coverage_pct = coverage_report.get('coverage_percentage', 0)
                logger.info(f"Coverage: {coverage_pct:.2f}%")
                logger.info(f"Total functions: {coverage_report.get('total_functions', 0)}")
                logger.info(f"Covered functions: {coverage_report.get('covered_functions', 0)}")
            
            return coverage_report
        except Exception as e:
            logger.error(f"Coverage analysis failed: {e}", exc_info=True)
            raise
    
    def identify_gaps(self) -> List[Dict[str, Any]]:
        """Identify coverage gaps with strategic analysis"""
        logger.info("Performing strategic coverage gap analysis...")

        try:
            # Get raw coverage data
            coverage_report = self.results.get('coverage_report', {})
            if not coverage_report:
                logger.warning("No coverage report available, running coverage analysis...")
                coverage_report = self.run_coverage_analysis()
                if not coverage_report:
                    logger.error("Could not obtain coverage data")
                    return []

            # Perform strategic coverage analysis
            strategic_analysis = analyze_coverage_strategically(coverage_report, self.config)

            gaps = strategic_analysis['gaps']
            insights = strategic_analysis['insights']

            # Log strategic insights
            logger.info(f"Strategic Analysis: {insights.get('total_gaps', 0)} total gaps, "
                       f"{insights.get('high_priority_gaps', 0)} high priority")

            # Log key insights
            risk_dist = insights.get('risk_distribution', {})
            if risk_dist.get('critical', 0) > 0:
                logger.warning(f"⚠️  {risk_dist['critical']} critical risk gaps identified")

            # Log strategic recommendations
            recommendations = insights.get('strategic_recommendations', [])
            if recommendations:
                logger.info("Strategic Recommendations:")
                for rec in recommendations[:3]:  # Show top 3
                    logger.info(f"  • {rec}")

            logger.info(f"Strategic gap analysis complete: {len(gaps)} gaps identified")
            return gaps

        except Exception as e:
            logger.error(f"Strategic gap identification failed: {e}", exc_info=True)
            # Fallback to basic gap identification
            logger.info("Falling back to basic gap identification...")
            try:
                existing_tests = []
                test_patterns = self.config.framework.test_patterns
                for pattern in test_patterns:
                    for test_file in self.config.test_dir.glob(pattern):
                        existing_tests.append(test_file.stem)

                gaps = self.coverage_analyzer.identify_test_gaps(existing_tests)
                logger.info(f"Basic gap identification: {len(gaps)} gaps found")
                return gaps
            except Exception as fallback_error:
                logger.error(f"Fallback gap identification also failed: {fallback_error}")
                return []
    
    def generate_test_vectors(self, gaps: List[Dict[str, Any]], use_llm: bool = None,
                             use_templates: bool = True, use_parallel: bool = False) -> List[TestVector]:
        """Generate test vectors for identified gaps using template-based generation with optional parallel processing"""
        logger.info("Generating test vectors...")

        if use_llm is None:
            use_llm = self.config.llm_enabled

        if use_parallel:
            return self._generate_parallel_vectors(gaps, use_templates)
        else:
            return self._generate_sequential_vectors(gaps, use_llm, use_templates)

    def _generate_sequential_vectors(self, gaps: List[Dict[str, Any]], use_llm: bool,
                                   use_templates: bool) -> List[TestVector]:
        """Generate test vectors sequentially (original implementation)"""
        generated = []
        template_generated = []
        llm_generated = []

        try:
            # Process gaps in batches
            batch_size = self.config.batch_size
            total_batches = (len(gaps) + batch_size - 1) // batch_size

            for i in range(0, len(gaps), batch_size):
                batch = gaps[i:i+batch_size]
                batch_num = i//batch_size + 1
                logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} gaps)...")

                for gap in batch:
                    try:
                        module_name = gap['module']
                        priority = gap.get('priority', 'medium')
                        if module_name in getattr(self.config, "integration_targets", []):
                            priority = "high"

                        module_vectors = []

                        # Try template-based generation first (faster, more reliable)
                        if use_templates:
                            template_vectors = self._generate_template_based_vectors(
                                module_name, gap, priority
                            )
                            module_vectors.extend(template_vectors)
                            template_generated.extend(template_vectors)

                        # Try LLM generation if enabled and we need more coverage
                        if use_llm and len(module_vectors) < 3:  # Generate at least 3 tests per module
                            try:
                                llm_vectors = self.llm_generator.generate_vectors_for_module(
                                    module_name,
                                    priority=priority
                                )
                                module_vectors.extend(llm_vectors)
                                llm_generated.extend(llm_vectors)
                            except Exception as llm_error:
                                logger.warning(f"LLM generation failed for {module_name}, "
                                             f"using template fallback: {llm_error}")

                        # Register all generated vectors
                        for vector in module_vectors:
                            self.registry.register(vector)
                            generated.append(vector)
                    except Exception as e:
                        logger.warning(f"Failed to generate vectors for {gap.get('module', 'unknown')}: {e}")
                        continue

            # Update results with generation statistics
            self.results['generated_vectors'] = [v.to_dict() for v in generated]
            self.results['generation_stats'] = {
                'total_generated': len(generated),
                'template_generated': len(template_generated),
                'llm_generated': len(llm_generated),
                'template_percentage': len(template_generated) / max(1, len(generated)) * 100,
                'llm_percentage': len(llm_generated) / max(1, len(generated)) * 100,
            }

            logger.info(f"Generated {len(generated)} test vectors "
                      f"({len(template_generated)} template-based, {len(llm_generated)} LLM-based)")

            return generated
        except Exception as e:
            logger.error(f"Test vector generation failed: {e}", exc_info=True)
            raise

    def _generate_parallel_vectors(self, gaps: List[Dict[str, Any]], use_templates: bool) -> List[TestVector]:
        """Generate test vectors using parallel processing"""
        logger.info("Using parallel generation engine...")

        try:
            # Extract module names from gaps
            module_names = [gap['module'] for gap in gaps]

            # Create priorities mapping
            priorities = {}
            for gap in gaps:
                module_name = gap['module']
                priority = gap.get('priority', 'medium')
                if module_name in getattr(self.config, "integration_targets", []):
                    priority = "high"
                priorities[module_name] = priority

            # Create parallel campaign
            campaign_id = self.parallel_engine.create_campaign(module_names, priorities)

            # Define generation function for parallel execution
            def generation_func(module_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
                """Generation function for parallel execution"""
                try:
                    priority = priorities.get(module_name, 'medium')
                    gap = next((g for g in gaps if g['module'] == module_name), {})

                    vectors = []
                    if use_templates:
                        template_vectors = self._generate_template_based_vectors(
                            module_name, gap, priority
                        )
                        vectors.extend(template_vectors)

                    return {
                        'vectors': vectors,
                        'metadata': {
                            'module': module_name,
                            'priority': priority,
                            'template_count': len(vectors)
                        }
                    }
                except Exception as e:
                    logger.error(f"Parallel generation failed for {module_name}: {e}")
                    return {'vectors': [], 'metadata': {'error': str(e)}}

            # Execute parallel campaign
            campaign_summary = self.parallel_engine.execute_campaign(campaign_id, generation_func)

            # Collect all generated vectors
            generated = []
            template_generated = []

            for result in campaign_summary.get('results', []):
                if result.get('success', False):
                    vectors = result.get('test_vectors', [])
                    generated.extend(vectors)
                    template_generated.extend(vectors)

                    # Register vectors
                    for vector in vectors:
                        self.registry.register(vector)

            # Update results with parallel generation statistics
            self.results['generated_vectors'] = [v.to_dict() for v in generated]
            self.results['generation_stats'] = {
                'total_generated': len(generated),
                'template_generated': len(template_generated),
                'parallel_campaign': campaign_summary
            }

            logger.info(f"Parallel generation completed: {len(generated)} test vectors generated")
            logger.info(f"Campaign duration: {campaign_summary.get('duration', 0):.2f}s")
            logger.info(f"Throughput: {campaign_summary.get('throughput', 0):.2f} tasks/sec")

            return generated

        except Exception as e:
            logger.error(f"Parallel generation failed: {e}", exc_info=True)
            # Fallback to sequential generation
            logger.info("Falling back to sequential generation...")
            return self._generate_sequential_vectors(gaps, False, use_templates)
    
    def save_results(self):
        """Save test results"""
        try:
            results_file = self.config.output_dir / "test_results.json"
            with open(results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            # Save registry
            registry_file = self.config.output_dir / "test_vectors.json"
            self.registry.save(registry_file)
            
            logger.info(f"Results saved to {self.config.output_dir}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}", exc_info=True)
            raise
    
    def save_generated_tests(self):
        """Save generated test code"""
        if self.config.save_generated_tests:
            generated_dir = self.config.output_dir / "generated_tests"
            self.llm_generator.save_generated_tests(generated_dir)
    
    def run_full_harness(self, use_llm: Optional[bool] = None,
                         focus_modules: Optional[List[str]] = None,
                         initialize: bool = True, use_parallel: bool = False) -> Dict[str, Any]:
        """
        Run full test harness workflow
        
        Args:
            use_llm: Whether to use LLM for test generation (overrides config)
            focus_modules: Optional list of module names to focus on
            initialize: Whether to initialize vector database first
        """
        logger.info("Starting unified test harness...")
        start_time = time.time()
        
        try:
            # Initialize if needed
            if initialize and self.config.use_vector_db:
                self.initialize()
            
            # 1. Run coverage analysis
            coverage_report = self.run_coverage_analysis()
            
            # 2. Identify gaps
            gaps = self.identify_gaps()
            
            # Filter by focus modules if specified
            if focus_modules:
                gaps = [g for g in gaps if g['module'] in focus_modules]
                logger.info(f"Filtered to {len(gaps)} focus modules")
            
            # 3. Generate test vectors
            vectors = self.generate_test_vectors(gaps, use_llm=use_llm, use_parallel=use_parallel)
            
            # 4. Save generated tests
            self.save_generated_tests()
            
            # 5. Save results
            self.save_results()
            
            # 6. Generate summary report
            self._generate_summary_report(gaps)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Test harness completed in {elapsed_time:.2f} seconds")
            
            return self.results
        except Exception as e:
            logger.error(f"Test harness failed: {e}", exc_info=True)
            raise
    
    def _generate_summary_report(self, gaps: List[Dict[str, Any]]):
        """Generate summary report"""
        report_file = self.config.output_dir / "HARNESS_SUMMARY.md"
        
        with open(report_file, 'w') as f:
            f.write("# Unified Test Harness Execution Summary\n\n")
            f.write(f"**Generated**: {self.results['run_timestamp']}\n\n")
            
            f.write("## Configuration\n\n")
            f.write(f"- **Source Root**: {self.config.source_root}\n")
            f.write(f"- **Test Directory**: {self.config.test_dir}\n")
            f.write(f"- **Framework**: {self.config.framework.test_framework}\n")
            f.write(f"- **LLM Enabled**: {self.config.llm_enabled}\n")
            f.write(f"- **Vector DB Enabled**: {self.config.use_vector_db}\n\n")
            
            f.write("## Coverage Report\n\n")
            coverage_pct = self.results['coverage_report'].get('coverage_percentage', 0)
            f.write(f"- **Overall Coverage**: {coverage_pct:.2f}%\n")
            f.write(f"- **Total Functions**: {self.results['coverage_report'].get('total_functions', 0)}\n")
            f.write(f"- **Covered Functions**: {self.results['coverage_report'].get('covered_functions', 0)}\n\n")
            
            f.write("## Coverage Gaps\n\n")
            if gaps:
                f.write("| Module | File Path | Uncovered Functions | Priority |\n")
                f.write("|--------|-----------|---------------------|----------|\n")
                for gap in gaps[:50]:  # Limit to first 50
                    uncovered = gap.get('uncovered_functions', [])
                    f.write(f"| {gap['module']} | {gap.get('file_path', 'N/A')} | "
                           f"{len(uncovered)} | {gap.get('priority', 'medium')} |\n")
            else:
                f.write("No coverage gaps identified.\n")
            f.write("\n")
            
            f.write("## Generated Test Vectors\n\n")
            vectors = self.results.get('generated_vectors', [])
            f.write(f"- **Total Vectors**: {len(vectors)}\n")
            if vectors:
                f.write("\n### Vector Summary\n\n")
                for vec in vectors[:20]:  # Show first 20
                    f.write(f"- **{vec.get('name', 'Unknown')}** ({vec.get('module_name', 'unknown')})\n")
                    f.write(f"  - Type: {vec.get('vector_type', 'unknown')}\n")
                    f.write(f"  - Priority: {vec.get('priority', 'medium')}\n")
                    f.write(f"  - Coverage Targets: {', '.join(vec.get('coverage_targets', []))}\n")
        
            logger.info(f"Summary report saved to {report_file}")

    def _generate_template_based_vectors(self, module_name: str, gap_info: Dict[str, Any],
                                       priority: str) -> List[TestVector]:
        """Generate test vectors using template-based approach"""
        vectors = []

        try:
            # Get module information
            module_file = gap_info.get('file_path', '')
            uncovered_functions = gap_info.get('uncovered_functions', [])

            # Determine language and framework from file extension and content
            language = self._detect_module_language(module_file)
            framework = self._detect_test_framework(language)

            # Generate vectors for uncovered functions
            for func_name in uncovered_functions[:5]:  # Limit to avoid overload
                try:
                    # Create code context for the function
                    code_context = self._analyze_function_context(module_name, func_name, module_file)

                    if not code_context:
                        continue

                    # Enhance analysis with plugins
                    plugin_analysis = self._analyze_with_plugins(code_context.code, module_name)
                    if plugin_analysis:
                        # Merge plugin analysis with code context
                        code_context.functions = plugin_analysis.get('functions', code_context.functions)
                        code_context.complexity = plugin_analysis.get('complexity', {}).get('average_complexity', code_context.complexity)
                        # Store additional plugin data
                        code_context.plugin_data = plugin_analysis

                        # Update language/framework if detected by plugins
                        if plugin_analysis.get('framework') and not framework:
                            framework = plugin_analysis['framework']

                    # Find best template
                    test_type = self._determine_test_type(code_context, priority)
                    template = self.template_engine.get_best_template(
                        language=language,
                        framework=framework,
                        test_type=test_type,
                        min_quality=0.5
                    )

                    if not template:
                        continue

                    # Apply template to generate test
                    generated_test = self.template_engine.apply_template(template, code_context)

                    if generated_test:
                        # Perform comprehensive quality validation (inspired by vector_revamp)
                        quality_report = self.template_engine.quality_validator.validate_test(
                            test_code=generated_test.test_code,
                            language=template.language,
                            framework=template.framework,
                            target_function=generated_test.target_function,
                            context={'target_complexity': code_context.complexity}
                        )

                        # Update generated test with validated quality score
                        validated_score = quality_report.overall_score

                        # Additional validation with plugins
                        plugin_validation = self._validate_with_plugins(
                            generated_test.test_code,
                            language,
                            framework,
                            plugin_analysis.get('domain') if plugin_analysis else None
                        )

                        # Adjust quality score based on plugin validation
                        if plugin_validation and 'overall_score' in plugin_validation:
                            combined_score = (validated_score + plugin_validation['overall_score']) / 2
                            validated_score = combined_score

                        generated_test.quality_score = validated_score

                        # Track quality metrics
                        template_id = template.template_id
                        if template_id not in self.template_engine.generation_quality_history:
                            self.template_engine.generation_quality_history[template_id] = []
                        self.template_engine.generation_quality_history[template_id].append(validated_score)

                        if generated_test.quality_score > self.config.template_quality_threshold:
                            # Convert to TestVector format
                            vector = self._convert_generated_test_to_vector(
                                generated_test, module_name, priority
                            )
                            if vector:
                                vectors.append(vector)

                except Exception as e:
                    logger.debug(f"Template generation failed for {func_name}: {e}")
                    continue

            # If no function-specific tests, generate module-level test
            if not vectors and uncovered_functions:
                vector = self._generate_module_level_template_test(
                    module_name, gap_info, language, framework, priority
                )
                if vector:
                    vectors.append(vector)

        except Exception as e:
            logger.warning(f"Template-based generation failed for {module_name}: {e}")

        return vectors

    def _analyze_function_context(self, module_name: str, func_name: str,
                                module_file: str) -> Optional[CodeContext]:
        """Analyze function to create code context for template application"""
        try:
            # This is a simplified analysis - in a full implementation,
            # this would use AST parsing or other code analysis techniques

            # Basic context creation
            context = CodeContext(
                module_name=module_name,
                function_name=func_name,
                function_signature=f"{func_name}(...)",  # Simplified
                parameters={},  # Would be populated from analysis
                return_type="Any",  # Would be determined from analysis
                decorators=[],
                docstring="",  # Would be extracted
                imports=[],  # Would be analyzed
                dependencies=[],  # Would be analyzed
                complexity=1,  # Basic complexity
                has_exceptions=False,  # Would be determined
                has_side_effects=False,  # Would be determined
                testability_score=0.7  # Basic score
            )

            return context

        except Exception as e:
            logger.debug(f"Function context analysis failed for {func_name}: {e}")
            return None

    def _determine_test_type(self, context: CodeContext, priority: str) -> str:
        """Determine appropriate test type based on context"""
        if priority == "high" or context.complexity > 3:
            return "integration"
        elif context.has_exceptions:
            return "edge_case"
        elif context.complexity > 1:
            return "unit"
        else:
            return "unit"

    def _detect_module_language(self, module_file: str) -> str:
        """Detect programming language from module file"""
        if module_file.endswith('.py'):
            return 'python'
        elif module_file.endswith(('.c', '.h')):
            return 'c'
        elif module_file.endswith('.rs'):
            return 'rust'
        else:
            return 'python'  # Default

    def _detect_test_framework(self, language: str) -> str:
        """Detect appropriate test framework for language"""
        if language == 'python':
            return 'pytest'  # Default for Python
        elif language == 'c':
            return 'unity'   # Default for C
        elif language == 'rust':
            return 'rust_builtin'
        else:
            return 'pytest'

    def _convert_generated_test_to_vector(self, generated_test, module_name: str,
                                        priority: str) -> Optional[TestVector]:
        """Convert generated test to TestVector format"""
        try:
            # Create test vector metadata
            metadata = {
                'generation_method': 'template_based',
                'template_used': generated_test.template_used,
                'quality_score': generated_test.quality_score,
                'coverage_estimate': generated_test.coverage_estimate,
                'target_function': generated_test.target_function,
                'language': generated_test.metadata.get('language'),
                'framework': generated_test.metadata.get('framework'),
            }

            # Create TestVector
            vector = TestVector(
                name=f"template_{module_name}_{generated_test.target_function}",
                module_name=module_name,
                vector_type=generated_test.test_type,
                priority=priority,
                coverage_targets=[generated_test.target_function],
                test_code=generated_test.test_code,
                metadata=metadata
            )

            return vector

        except Exception as e:
            logger.debug(f"Vector conversion failed: {e}")
            return None

    def _generate_module_level_template_test(self, module_name: str, gap_info: Dict[str, Any],
                                           language: str, framework: str, priority: str) -> Optional[TestVector]:
        """Generate a module-level test when function-specific tests aren't available"""
        try:
            # Create a basic module-level test
            test_code = self._create_basic_module_test(module_name, language, framework)

            metadata = {
                'generation_method': 'template_module_level',
                'quality_score': 0.5,
                'coverage_estimate': 0.3,
                'language': language,
                'framework': framework,
            }

            vector = TestVector(
                name=f"template_{module_name}_module_test",
                module_name=module_name,
                vector_type="unit",
                priority=priority,
                coverage_targets=["module_import"],
                test_code=test_code,
                metadata=metadata
            )

            return vector

        except Exception as e:
            logger.debug(f"Module-level test generation failed: {e}")
            return None

    def _create_basic_module_test(self, module_name: str, language: str, framework: str) -> str:
        """Create a basic module import test"""
        if language == 'python':
            if framework == 'pytest':
                return f'''def test_{module_name}_import():
    """Test that {module_name} module can be imported"""
    try:
        import {module_name}
        assert {module_name} is not None
    except ImportError:
        pytest.fail(f"Failed to import {module_name}")
'''
            else:  # unittest
                return f'''def test_{module_name}_import(self):
    """Test that {module_name} module can be imported"""
    try:
        import {module_name}
        self.assertIsNotNone({module_name})
    except ImportError:
        self.fail(f"Failed to import {module_name}")
'''

        elif language == 'c':
            if framework == 'unity':
                return f'''void test_{module_name}_compilation(void) {{
    // Test that {module_name} compiles and links
    TEST_ASSERT_TRUE(true);  // Basic compilation test
}}
'''
            elif framework == 'cmocka':
                return f'''static void test_{module_name}_compilation(void **state) {{
    // Test that {module_name} compiles and links
    assert_true(true);  // Basic compilation test
}}
'''

        elif language == 'rust':
            return f'''#[test]
fn test_{module_name}_compilation() {{
    // Test that {module_name} compiles
    assert!(true);  // Basic compilation test
}}
'''

        else:
            return f'# Basic test for {module_name}'

    def _analyze_with_plugins(self, code: str, module_name: str) -> Optional[Dict[str, Any]]:
        """Analyze code using available plugins."""
        try:
            language = self.config.language
            file_path = Path(module_name.replace('.', '/'))  # Convert module name to path
            if not file_path.suffix:
                file_path = file_path.with_suffix('.py')  # Assume Python

            return self.plugin_orchestrator.analyze_code_with_plugins(code, language, file_path)
        except Exception as e:
            logger.debug(f"Plugin analysis failed for {module_name}: {e}")
            return None

    def _validate_with_plugins(self, test_code: str, language: str, framework: str = None,
                              domain: str = None) -> Optional[Dict[str, Any]]:
        """Validate test code using plugins."""
        try:
            return self.plugin_orchestrator.validate_with_plugins(test_code, language, framework, domain)
        except Exception as e:
            logger.debug(f"Plugin validation failed: {e}")
            return None


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Test Harness Runner")
    parser.add_argument("--source-root", type=Path, default=Path.cwd(),
                       help="Source code root directory")
    parser.add_argument("--test-dir", type=Path, default=None,
                       help="Test directory (default: source_root/tests)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: test_dir/harness_output)")
    parser.add_argument("--project-type", type=str, choices=["standard", "src_layout", "modules_layout"],
                       default="standard", help="Project structure type")
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for test generation")
    parser.add_argument("--llm-provider", type=str, default="openai",
                       help="LLM provider (openai, anthropic)")
    parser.add_argument("--llm-api-key", type=str, default=None,
                       help="LLM API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)")
    parser.add_argument("--init", action="store_true",
                       help="Initialize vector database")
    parser.add_argument("--coverage-only", action="store_true",
                       help="Only run coverage analysis")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate tests (skip coverage)")
    
    args = parser.parse_args()
    
    # Determine paths
    source_root = args.source_root.resolve()
    test_dir = args.test_dir or (source_root / "tests")
    output_dir = args.output_dir or (test_dir / "harness_output")
    
    # Create configuration
    config = HarnessConfig.create_for_project(source_root, args.project_type)
    config.test_dir = test_dir
    config.output_dir = output_dir
    config.vector_db_path = output_dir / "vector_db"
    
    if args.use_llm:
        config.llm_enabled = True
        config.llm_provider = args.llm_provider
        config.llm_api_key = args.llm_api_key
    
    # Create runner
    runner = TestHarnessRunner(config)
    
    if args.init:
        runner.initialize()
    elif args.coverage_only:
        runner.run_coverage_analysis()
        runner.save_results()
    elif args.generate_only:
        runner.generate_test_vectors([])
        runner.save_generated_tests()
        runner.save_results()
    else:
        results = runner.run_full_harness(use_llm=args.use_llm, initialize=args.init)
        logger.info("\nTest harness completed!")
        logger.info(f"Coverage: {results['coverage_report'].get('coverage_percentage', 0):.2f}%")
        logger.info(f"Generated vectors: {len(results['generated_vectors'])}")


if __name__ == "__main__":
    main()
