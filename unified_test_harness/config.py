"""
Configuration System for Unified Test Harness

Supports different frameworks and project structures.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class FrameworkConfig:
    """Framework-specific configuration"""
    name: str
    test_framework: str  # 'pytest', 'unittest', 'nose', etc.
    source_patterns: List[str]  # Patterns to identify source files
    test_patterns: List[str]  # Patterns to identify test files
    import_prefix: str  # How to import modules (e.g., 'src.', 'modules.', '')
    test_import_prefix: str  # How to import in tests
    coverage_source: str  # What to pass to coverage tool
    test_command: List[str]  # Command to run tests
    coverage_command: List[str]  # Command to run coverage


@dataclass
class HarnessConfig:
    """Main configuration for the test harness"""
    
    # Project paths
    source_root: Path
    test_dir: Path
    output_dir: Path
    vector_db_path: Path
    
    # Framework configuration
    framework: FrameworkConfig
    
    # LLM configuration
    llm_enabled: bool = False
    llm_provider: str = "openai"  # 'openai', 'anthropic', 'local', etc.
    llm_api_key: Optional[str] = None
    llm_model: str = "gpt-4"
    
    # Coverage configuration
    coverage_threshold: float = 0.8
    coverage_minimum: float = 0.0
    
    # Generation configuration
    batch_size: int = 50
    max_tests_per_module: int = 100
    
    # Vector database configuration
    use_vector_db: bool = True
    embedding_model: str = "default"  # 'default', 'codebert', 'custom'
    
    # Test generation configuration
    generate_unit_tests: bool = True
    generate_integration_tests: bool = True
    generate_edge_cases: bool = True
    generate_error_tests: bool = True
    
    # Output configuration
    save_generated_tests: bool = True
    test_output_format: str = "pytest"  # 'pytest', 'unittest', 'raw'
    
    # Advanced options
    parallel_generation: bool = False
    similarity_threshold: float = 0.7
    
    @classmethod
    def create_default(cls, source_root: Path, test_dir: Path, output_dir: Path) -> 'HarnessConfig':
        """Create default configuration"""
        return cls(
            source_root=source_root,
            test_dir=test_dir,
            output_dir=output_dir,
            vector_db_path=output_dir / "vector_db",
            framework=FrameworkConfig(
                name="default",
                test_framework="pytest",
                source_patterns=["**/*.py"],
                test_patterns=["test_*.py", "*_test.py"],
                import_prefix="",
                test_import_prefix="",
                coverage_source=".",
                test_command=["pytest"],
                coverage_command=["pytest", "--cov", ".", "--cov-report=xml"],
            )
        )
    
    @classmethod
    def create_for_project(cls, source_root: Path, project_type: str = "standard", language: str = "python") -> 'HarnessConfig':
        """Create configuration for common project types
        
        Args:
            source_root: Root directory of the project
            project_type: Project structure type ("standard", "src_layout", "modules_layout", "c_project", "rust_project")
            language: Primary language ("python", "c", "rust")
        """
        test_dir = source_root / "tests"
        output_dir = source_root / "tests" / "harness_output"
        
        if language == "c" or project_type == "c_project":
            # C project structure
            framework = FrameworkConfig(
                name="c_project",
                test_framework="unity",  # Unity test framework
                source_patterns=["src/**/*.c", "src/**/*.h", "**/*.c", "**/*.h"],
                test_patterns=["tests/**/test_*.c", "tests/**/*_test.c"],
                import_prefix="",
                test_import_prefix="",
                coverage_source=".",
                test_command=["make", "test"],  # Assumes Makefile
                coverage_command=["make", "coverage"],  # Assumes Makefile with coverage target
            )
        elif language == "rust" or project_type == "rust_project":
            # Rust project structure
            framework = FrameworkConfig(
                name="rust_project",
                test_framework="cargo",
                source_patterns=["src/**/*.rs"],
                test_patterns=["tests/**/*.rs", "**/*test*.rs"],
                import_prefix="",
                test_import_prefix="",
                coverage_source="src",
                test_command=["cargo", "test"],
                coverage_command=["cargo", "tarpaulin", "--out", "Xml"],
            )
        elif project_type == "src_layout":
            # src/ structure (Python)
            framework = FrameworkConfig(
                name="src_layout",
                test_framework="pytest",
                source_patterns=["src/**/*.py"],
                test_patterns=["tests/**/test_*.py"],
                import_prefix="src.",
                test_import_prefix="src.",
                coverage_source="src",
                test_command=["pytest", "tests"],
                coverage_command=["pytest", "tests", "--cov=src", "--cov-report=xml"],
            )
        elif project_type == "modules_layout":
            # modules/ structure (Python)
            framework = FrameworkConfig(
                name="modules_layout",
                test_framework="pytest",
                source_patterns=["modules/**/*.py"],
                test_patterns=["tests/**/test_*.py"],
                import_prefix="modules.",
                test_import_prefix="modules.",
                coverage_source="modules",
                test_command=["pytest", "tests"],
                coverage_command=["pytest", "tests", "--cov=modules", "--cov-report=xml"],
            )
        else:
            # Standard flat structure (Python)
            framework = FrameworkConfig(
                name="standard",
                test_framework="pytest",
                source_patterns=["**/*.py"],
                test_patterns=["test_*.py", "*_test.py"],
                import_prefix="",
                test_import_prefix="",
                coverage_source=".",
                test_command=["pytest"],
                coverage_command=["pytest", "--cov", ".", "--cov-report=xml"],
            )
        
        return cls(
            source_root=source_root,
            test_dir=test_dir,
            output_dir=output_dir,
            vector_db_path=output_dir / "vector_db",
            framework=framework,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'source_root': str(self.source_root),
            'test_dir': str(self.test_dir),
            'output_dir': str(self.output_dir),
            'vector_db_path': str(self.vector_db_path),
            'framework': {
                'name': self.framework.name,
                'test_framework': self.framework.test_framework,
                'source_patterns': self.framework.source_patterns,
                'test_patterns': self.framework.test_patterns,
                'import_prefix': self.framework.import_prefix,
                'test_import_prefix': self.framework.test_import_prefix,
                'coverage_source': self.framework.coverage_source,
                'test_command': self.framework.test_command,
                'coverage_command': self.framework.coverage_command,
            },
            'llm_enabled': self.llm_enabled,
            'llm_provider': self.llm_provider,
            'llm_model': self.llm_model,
            'coverage_threshold': self.coverage_threshold,
            'batch_size': self.batch_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HarnessConfig':
        """Create from dictionary"""
        framework_data = data['framework']
        framework = FrameworkConfig(
            name=framework_data['name'],
            test_framework=framework_data['test_framework'],
            source_patterns=framework_data['source_patterns'],
            test_patterns=framework_data['test_patterns'],
            import_prefix=framework_data['import_prefix'],
            test_import_prefix=framework_data['test_import_prefix'],
            coverage_source=framework_data['coverage_source'],
            test_command=framework_data['test_command'],
            coverage_command=framework_data['coverage_command'],
        )
        
        return cls(
            source_root=Path(data['source_root']),
            test_dir=Path(data['test_dir']),
            output_dir=Path(data['output_dir']),
            vector_db_path=Path(data['vector_db_path']),
            framework=framework,
            llm_enabled=data.get('llm_enabled', False),
            llm_provider=data.get('llm_provider', 'openai'),
            llm_model=data.get('llm_model', 'gpt-4'),
            coverage_threshold=data.get('coverage_threshold', 0.8),
            batch_size=data.get('batch_size', 50),
        )
