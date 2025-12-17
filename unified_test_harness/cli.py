#!/usr/bin/env python3
"""
Command-line interface for Unified Test Harness
"""

import sys
import json
import logging
from pathlib import Path
from .harness_runner import TestHarnessRunner
from .config import HarnessConfig
from .language_parser import LanguageParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


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
                       choices=["standard", "src_layout", "modules_layout", "c_project", "rust_project"],
                       default=None,
                       help="Project structure type (auto-detected if not specified)")
    parser.add_argument("--language", type=str,
                       choices=["python", "c", "rust"],
                       default=None,
                       help="Primary language (auto-detected if not specified)")
    
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
    
    # Configuration file support
    parser.add_argument("--config", type=Path, default=None,
                       help="Path to JSON configuration file (overrides command-line arguments)")
    
    args = parser.parse_args()
    
    # Load configuration from file if provided
    config_data = {}
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            sys.exit(1)
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            logger.info(f"Loaded configuration from: {config_path}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error reading configuration file: {e}")
            sys.exit(1)
    
    # Override command-line arguments with config file values
    if config_data:
        args.source_root = Path(config_data.get('source_root', args.source_root))
        args.test_dir = Path(config_data.get('test_dir')) if config_data.get('test_dir') else args.test_dir
        args.output_dir = Path(config_data.get('output_dir')) if config_data.get('output_dir') else args.output_dir
        args.project_type = config_data.get('project_type', args.project_type)
        args.language = config_data.get('language', args.language)
        args.use_llm = config_data.get('llm_enabled', args.use_llm)
        args.llm_provider = config_data.get('llm_provider', args.llm_provider)
        args.llm_model = config_data.get('llm_model', args.llm_model)
        args.batch_size = config_data.get('batch_size', args.batch_size)
        args.coverage_threshold = config_data.get('coverage_threshold', args.coverage_threshold)
        args.no_vector_db = not config_data.get('use_vector_db', not args.no_vector_db)
        if 'modules' in config_data:
            args.modules = config_data['modules']
    
    # Determine paths
    source_root = args.source_root.resolve()
    test_dir = args.test_dir or (source_root / "tests")
    output_dir = args.output_dir or (test_dir / "harness_output")
    
    # Auto-detect language if not specified
    language = args.language
    if not language:
        lang_parser = LanguageParser()
        # Check for common files
        if (source_root / "Cargo.toml").exists():
            language = "rust"
        elif (source_root / "Makefile").exists() or list(source_root.glob("*.c")):
            language = "c"
        elif list(source_root.glob("*.py")):
            language = "python"
        else:
            language = "python"  # Default
    
    # Auto-detect project type if not specified
    project_type = args.project_type
    if not project_type:
        if language == "rust":
            project_type = "rust_project"
        elif language == "c":
            project_type = "c_project"
        else:
            # Check for Python project structure
            if (source_root / "src").exists():
                project_type = "src_layout"
            elif (source_root / "modules").exists():
                project_type = "modules_layout"
            else:
                project_type = "standard"
    
    logger.info(f"Detected language: {language}")
    logger.info(f"Using project type: {project_type}")
    
    # Create configuration
    config = HarnessConfig.create_for_project(source_root, project_type, language)
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
            logger.info("\n" + "=" * 70)
            logger.info("Test Harness Completed Successfully!")
            logger.info("=" * 70)
            logger.info(f"Coverage: {results['coverage_report'].get('coverage_percentage', 0):.2f}%")
            logger.info(f"Generated vectors: {len(results['generated_vectors'])}")
            logger.info(f"Results saved to: {config.output_dir}")
            logger.info("=" * 70)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nError: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
