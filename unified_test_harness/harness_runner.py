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
                logger.info("\n[1/2] Embedding codebase...")
                self.code_embedder.embed_codebase()
                
                logger.info("\n[2/2] Embedding test templates...")
                self.code_embedder.embed_existing_tests()
            else:
                logger.warning("\nVector database disabled, skipping embedding")
            
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
        """Identify coverage gaps"""
        logger.info("Identifying coverage gaps...")
        
        try:
            # Get existing test files
            existing_tests = []
            test_patterns = self.config.framework.test_patterns
            for pattern in test_patterns:
                for test_file in self.config.test_dir.glob(pattern):
                    existing_tests.append(test_file.stem)
            
            # Identify gaps
            gaps = self.coverage_analyzer.identify_test_gaps(existing_tests)
            
            # Also identify zero-coverage modules
            zero_coverage = self.coverage_analyzer.identify_zero_coverage_modules(
                threshold=0.01
            )
            
            for module_info in zero_coverage:
                if module_info['module'] not in [g['module'] for g in gaps]:
                    gaps.append({
                        'module': module_info['module'],
                        'file_path': module_info['file_path'],
                        'coverage': module_info['coverage'],
                        'priority': module_info['priority'],
                        'uncovered_functions': [],
                        'type': 'zero_coverage'
                    })
            
            logger.info(f"Found {len(gaps)} coverage gaps")
            return gaps
        except Exception as e:
            logger.error(f"Gap identification failed: {e}", exc_info=True)
            raise
    
    def generate_test_vectors(self, gaps: List[Dict[str, Any]], use_llm: bool = None) -> List[TestVector]:
        """Generate test vectors for identified gaps"""
        logger.info("Generating test vectors...")
        
        if use_llm is None:
            use_llm = self.config.llm_enabled
        
        generated = []
        
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
                        
                        vectors = self.llm_generator.generate_vectors_for_module(
                            module_name,
                            priority=priority
                        )
                        
                        for vector in vectors:
                            self.registry.register(vector)
                            generated.append(vector)
                    except Exception as e:
                        logger.warning(f"Failed to generate vectors for {gap.get('module', 'unknown')}: {e}")
                        continue
            
            self.results['generated_vectors'] = [v.to_dict() for v in generated]
            logger.info(f"Generated {len(generated)} test vectors")
            
            return generated
        except Exception as e:
            logger.error(f"Test vector generation failed: {e}", exc_info=True)
            raise
    
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
                         initialize: bool = True) -> Dict[str, Any]:
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
            vectors = self.generate_test_vectors(gaps, use_llm=use_llm)
            
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
