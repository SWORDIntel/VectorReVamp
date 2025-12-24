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


def _load_plugins_for_runner(runner, plugin_dirs):
    """Load plugins for the test runner."""
    try:
        # Add default plugin directories
        default_dirs = [
            Path(__file__).parent / "plugins",
            Path.cwd() / "plugins",
        ]

        # Add user-specified directories
        for plugin_dir in plugin_dirs:
            default_dirs.append(plugin_dir)

        # Load plugins from each directory
        for plugin_dir in default_dirs:
            if plugin_dir.exists():
                runner.plugin_loader.add_plugin_directory(plugin_dir)

        # Discover and load plugins
        plugin_files = runner.plugin_loader.discover_plugins()
        for plugin_file in plugin_files:
            runner.plugin_loader.load_plugin(plugin_file)

        loaded_count = len(runner.plugin_registry.list_plugins())
        logger.info(f"Loaded {loaded_count} plugins")

    except Exception as e:
        logger.error(f"Plugin loading failed: {e}")


def list_available_plugins():
    """List all available plugins."""
    from .harness_runner import TestHarnessRunner
    from .config import HarnessConfig

    # Create a minimal config and runner for plugin access
    config = HarnessConfig.create_minimal()
    runner = TestHarnessRunner(config)

    # Load plugins
    _load_plugins_for_runner(runner, [])

    # Display plugins
    plugins = runner.plugin_registry.list_plugins()

    if not plugins:
        print("No plugins found.")
        return

    print("Available Plugins:")
    print("=" * 50)

    for plugin in plugins:
        print(f"ID: {plugin['id']}")
        print(f"Name: {plugin['name']}")
        print(f"Type: {plugin['type']}")
        print(f"Languages: {', '.join(plugin['languages'])}")
        print(f"Frameworks: {', '.join(plugin['frameworks'])}")
        print(f"Domains: {', '.join(plugin['domains'])}")
        print(f"Quality Score: {plugin['quality_score']:.2f}")
        print(f"Status: {'Loaded' if plugin['loaded'] else 'Not Loaded'}")
        print("-" * 30)


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

    # Template options
    parser.add_argument("--use-templates", action="store_true", default=True,
                       help="Use template-based generation (default: enabled)")
    parser.add_argument("--no-templates", action="store_true",
                       help="Disable template-based generation")
    parser.add_argument("--template-quality-threshold", type=float, default=0.6,
                       help="Minimum template quality score (default: 0.6)")
    parser.add_argument("--save-templates", type=Path, default=None,
                       help="Save extracted templates to JSON file")
    parser.add_argument("--load-templates", type=Path, default=None,
                       help="Load templates from JSON file")

    # Parallel processing options
    parser.add_argument("--parallel", action="store_true",
                       help="Enable parallel test generation")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (auto-detected if not specified)")

    # Plugin system options
    parser.add_argument("--plugin-dir", type=Path, action="append",
                       help="Directory to search for plugins (can be specified multiple times)")
    parser.add_argument("--list-plugins", action="store_true",
                       help="List available plugins and exit")
    parser.add_argument("--enable-plugins", action="store_true", default=True,
                       help="Enable plugin system (default: enabled)")
    parser.add_argument("--disable-plugins", action="store_true",
                       help="Disable plugin system")
    
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

    # Handle special commands
    if args.list_plugins:
        list_available_plugins()
        return

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

        # Template configuration
        args.use_templates = config_data.get('use_templates', args.use_templates)
        args.template_quality_threshold = config_data.get('template_quality_threshold', args.template_quality_threshold)
        if 'save_templates' in config_data:
            args.save_templates = Path(config_data['save_templates'])
        if 'load_templates' in config_data:
            args.load_templates = Path(config_data['load_templates'])

        # Parallel configuration
        args.parallel = config_data.get('parallel', args.parallel)
        args.max_workers = config_data.get('max_workers', args.max_workers)

        # Plugin configuration
        if 'plugin_dirs' in config_data:
            args.plugin_dir = [Path(d) for d in config_data['plugin_dirs']]
        args.enable_plugins = config_data.get('enable_plugins', args.enable_plugins)
        if args.disable_plugins:
            args.enable_plugins = False
    
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

    # Configure template engine
    if args.no_templates:
        args.use_templates = False
    config.use_templates = args.use_templates
    config.template_quality_threshold = args.template_quality_threshold

    # Handle template file operations
    if args.save_templates:
        config.save_templates_path = args.save_templates
    if args.load_templates:
        config.load_templates_path = args.load_templates
        if args.load_templates.exists():
            logger.info(f"Loading templates from: {args.load_templates}")
            runner.template_engine.load_templates(args.load_templates)
        else:
            logger.warning(f"Template file not found: {args.load_templates}")

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
            # Load plugins if enabled
            if args.enable_plugins:
                _load_plugins_for_runner(runner, args.plugin_dir or [])

            results = runner.run_full_harness(
                use_llm=args.use_llm,
                focus_modules=args.modules,
                initialize=args.init,
                use_parallel=args.parallel
            )
            # Save templates if requested
            if hasattr(config, 'save_templates_path') and config.save_templates_path:
                logger.info(f"Saving templates to: {config.save_templates_path}")
                runner.template_engine.save_templates(config.save_templates_path)

            logger.info("\n" + "=" * 70)
            logger.info("Test Harness Completed Successfully!")
            logger.info("=" * 70)
            logger.info(f"Coverage: {results['coverage_report'].get('coverage_percentage', 0):.2f}%")
            logger.info(f"Generated vectors: {len(results['generated_vectors'])}")

            # Show generation statistics if available
            if 'generation_stats' in results:
                stats = results['generation_stats']
                logger.info(f"Template-based: {stats.get('template_generated', 0)} "
                          f"({stats.get('template_percentage', 0):.1f}%)")
                logger.info(f"LLM-based: {stats.get('llm_generated', 0)} "
                          f"({stats.get('llm_percentage', 0):.1f}%)")

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
