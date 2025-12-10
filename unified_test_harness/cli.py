#!/usr/bin/env python3
"""
Command-line interface for Unified Test Harness
"""

import sys
from pathlib import Path
from .harness_runner import TestHarnessRunner
from .config import HarnessConfig


def main():
    """Main CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Unified Test Harness - Framework-agnostic test generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Initialize harness for standard project
  unified-test-harness --init
  
  # Run full harness with LLM
  unified-test-harness --use-llm --llm-provider openai
  
  # Run coverage analysis only
  unified-test-harness --coverage-only
  
  # Generate tests for specific modules
  unified-test-harness --modules module1 module2
  
  # Use custom project structure
  unified-test-harness --project-type src_layout --init
        """
    )
    
    parser.add_argument("--source-root", type=Path, default=Path.cwd(),
                       help="Source code root directory (default: current directory)")
    parser.add_argument("--test-dir", type=Path, default=None,
                       help="Test directory (default: source_root/tests)")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: test_dir/harness_output)")
    parser.add_argument("--project-type", type=str, 
                       choices=["standard", "src_layout", "modules_layout"],
                       default="standard", 
                       help="Project structure type (default: standard)")
    
    # LLM options
    parser.add_argument("--use-llm", action="store_true",
                       help="Use LLM for test generation")
    parser.add_argument("--llm-provider", type=str, 
                       choices=["openai", "anthropic"],
                       default="openai",
                       help="LLM provider (default: openai)")
    parser.add_argument("--llm-api-key", type=str, default=None,
                       help="LLM API key (or set OPENAI_API_KEY/ANTHROPIC_API_KEY env var)")
    parser.add_argument("--llm-model", type=str, default="gpt-4",
                       help="LLM model name (default: gpt-4)")
    
    # Actions
    parser.add_argument("--init", action="store_true",
                       help="Initialize vector database")
    parser.add_argument("--coverage-only", action="store_true",
                       help="Only run coverage analysis")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate tests (skip coverage)")
    
    # Generation options
    parser.add_argument("--batch-size", type=int, default=50,
                       help="Batch size for test generation (default: 50)")
    parser.add_argument("--modules", nargs="+", default=None,
                       help="Focus on specific modules")
    parser.add_argument("--no-vector-db", action="store_true",
                       help="Disable vector database (faster, less intelligent)")
    
    # Coverage options
    parser.add_argument("--coverage-threshold", type=float, default=0.8,
                       help="Coverage threshold (default: 0.8)")
    
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
    config.batch_size = args.batch_size
    config.coverage_threshold = args.coverage_threshold
    config.use_vector_db = not args.no_vector_db
    
    if args.use_llm:
        config.llm_enabled = True
        config.llm_provider = args.llm_provider
        config.llm_api_key = args.llm_api_key
        config.llm_model = args.llm_model
    
    # Create runner
    runner = TestHarnessRunner(config)
    
    try:
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
            results = runner.run_full_harness(
                use_llm=args.use_llm,
                focus_modules=args.modules,
                initialize=args.init
            )
            print("\n" + "=" * 70)
            print("Test Harness Completed Successfully!")
            print("=" * 70)
            print(f"Coverage: {results['coverage_report'].get('coverage_percentage', 0):.2f}%")
            print(f"Generated vectors: {len(results['generated_vectors'])}")
            print(f"Results saved to: {config.output_dir}")
            print("=" * 70)
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
