"""
Extensible Plugin Architecture & Domain Intelligence for VectorReVamp

Modular plugin system allowing domain experts to contribute specialized
testing intelligence for different languages, frameworks, and problem domains.
Inspired by vector_revamp's domain-specific data generation patterns.
"""

import os
import sys
import json
import hashlib
import logging
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable, Type, Protocol, runtime_checkable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import importlib.util
import inspect

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata and capabilities."""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    license: str

    # Capabilities
    supported_languages: List[str] = field(default_factory=list)
    supported_frameworks: List[str] = field(default_factory=list)
    supported_domains: List[str] = field(default_factory=list)

    # Plugin properties
    plugin_type: str = "generic"  # 'language', 'framework', 'domain', 'tool'
    priority: int = 50  # 1-100, higher = more preferred
    requires_dependencies: List[str] = field(default_factory=list)

    # Quality and security
    security_rating: str = "unknown"  # 'verified', 'trusted', 'unknown', 'suspicious'
    quality_score: float = 0.5  # 0.0 to 1.0
    last_updated: str = ""

    # Runtime properties
    is_loaded: bool = False
    load_path: Optional[Path] = None


@dataclass
class PluginCapabilities:
    """Detailed plugin capabilities."""
    code_analysis: bool = False
    test_generation: bool = False
    quality_validation: bool = False
    coverage_analysis: bool = False
    template_provision: bool = False
    domain_intelligence: bool = False

    # Specialized capabilities
    language_parsing: bool = False
    framework_detection: bool = False
    domain_modeling: bool = False
    security_analysis: bool = False
    performance_optimization: bool = False


@runtime_checkable
class PluginInterface(Protocol):
    """Plugin interface protocol."""

    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata."""
        ...

    @property
    def capabilities(self) -> PluginCapabilities:
        """Plugin capabilities."""
        ...

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin with configuration."""
        ...

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        ...


class BasePlugin(ABC):
    """Base class for all plugins."""

    def __init__(self, metadata: PluginMetadata, capabilities: PluginCapabilities):
        self._metadata = metadata
        self._capabilities = capabilities
        self._config: Dict[str, Any] = {}
        self._initialized = False

    @property
    def metadata(self) -> PluginMetadata:
        return self._metadata

    @property
    def capabilities(self) -> PluginCapabilities:
        return self._capabilities

    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize plugin with configuration."""
        try:
            self._config = config
            self._initialized = True
            logger.info(f"Plugin {self.metadata.plugin_id} initialized")
            return True
        except Exception as e:
            logger.error(f"Plugin {self.metadata.plugin_id} initialization failed: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup plugin resources."""
        self._initialized = False
        logger.info(f"Plugin {self.metadata.plugin_id} cleaned up")

    def is_capable(self, capability: str) -> bool:
        """Check if plugin has specific capability."""
        return getattr(self.capabilities, capability, False)

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self._config.get(key, default)


class LanguagePlugin(BasePlugin):
    """Plugin for language-specific testing intelligence."""

    @abstractmethod
    def parse_code(self, code: str, file_path: Path) -> Dict[str, Any]:
        """Parse code and extract language-specific information."""
        pass

    @abstractmethod
    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function definitions from code."""
        pass

    @abstractmethod
    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity."""
        pass

    @abstractmethod
    def generate_language_specific_tests(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate language-specific test patterns."""
        pass


class FrameworkPlugin(BasePlugin):
    """Plugin for testing framework intelligence."""

    @abstractmethod
    def detect_framework(self, project_root: Path) -> Optional[str]:
        """Detect testing framework in project."""
        pass

    @abstractmethod
    def generate_framework_tests(self, functions: List[Dict[str, Any]],
                               framework: str) -> List[Dict[str, Any]]:
        """Generate framework-specific test code."""
        pass

    @abstractmethod
    def validate_framework_usage(self, test_code: str, framework: str) -> Dict[str, Any]:
        """Validate correct framework usage."""
        pass

    @abstractmethod
    def get_framework_patterns(self, framework: str) -> Dict[str, Any]:
        """Get framework-specific test patterns."""
        pass


class DomainPlugin(BasePlugin):
    """Plugin for domain-specific testing intelligence."""

    @abstractmethod
    def analyze_domain(self, code: str, functions: List[Dict[str, Any]]) -> str:
        """Analyze code to determine domain context."""
        pass

    @abstractmethod
    def get_domain_patterns(self, domain: str) -> Dict[str, Any]:
        """Get domain-specific test patterns."""
        pass

    @abstractmethod
    def generate_domain_tests(self, functions: List[Dict[str, Any]],
                            domain: str) -> List[Dict[str, Any]]:
        """Generate domain-specific tests."""
        pass

    @abstractmethod
    def validate_domain_correctness(self, test_code: str, domain: str) -> Dict[str, Any]:
        """Validate domain-specific correctness."""
        pass


class ToolPlugin(BasePlugin):
    """Plugin for external tool integration."""

    @abstractmethod
    def check_tool_availability(self) -> bool:
        """Check if external tool is available."""
        pass

    @abstractmethod
    def execute_tool(self, code: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Execute external tool on code."""
        pass

    @abstractmethod
    def parse_tool_output(self, output: str) -> Dict[str, Any]:
        """Parse tool output into structured data."""
        pass

    @abstractmethod
    def integrate_results(self, tool_results: Dict[str, Any],
                         existing_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate tool results with existing analysis."""
        pass


class PluginRegistry:
    """Registry for managing plugins."""

    def __init__(self):
        self._plugins: Dict[str, BasePlugin] = {}
        self._plugin_types: Dict[str, List[str]] = {
            'language': [],
            'framework': [],
            'domain': [],
            'tool': []
        }
        self._capabilities_index: Dict[str, List[str]] = {}
        self._language_index: Dict[str, List[str]] = {}
        self._framework_index: Dict[str, List[str]] = {}
        self._domain_index: Dict[str, List[str]] = {}

    def register_plugin(self, plugin: BasePlugin) -> bool:
        """Register a plugin in the registry."""
        plugin_id = plugin.metadata.plugin_id

        if plugin_id in self._plugins:
            logger.warning(f"Plugin {plugin_id} already registered")
            return False

        try:
            self._plugins[plugin_id] = plugin

            # Update type index
            plugin_type = plugin.metadata.plugin_type
            if plugin_type in self._plugin_types:
                self._plugin_types[plugin_type].append(plugin_id)

            # Update capability index
            for cap_name in PluginCapabilities.__annotations__.keys():
                if getattr(plugin.capabilities, cap_name, False):
                    if cap_name not in self._capabilities_index:
                        self._capabilities_index[cap_name] = []
                    self._capabilities_index[cap_name].append(plugin_id)

            # Update language/framework/domain indexes
            for lang in plugin.metadata.supported_languages:
                if lang not in self._language_index:
                    self._language_index[lang] = []
                self._language_index[lang].append(plugin_id)

            for framework in plugin.metadata.supported_frameworks:
                if framework not in self._framework_index:
                    self._framework_index[framework] = []
                self._framework_index[framework].append(plugin_id)

            for domain in plugin.metadata.supported_domains:
                if domain not in self._domain_index:
                    self._domain_index[domain] = []
                self._domain_index[domain].append(plugin_id)

            logger.info(f"Plugin {plugin_id} registered successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_id}: {e}")
            return False

    def unregister_plugin(self, plugin_id: str) -> bool:
        """Unregister a plugin."""
        if plugin_id not in self._plugins:
            return False

        plugin = self._plugins[plugin_id]

        # Remove from type index
        plugin_type = plugin.metadata.plugin_type
        if plugin_type in self._plugin_types and plugin_id in self._plugin_types[plugin_type]:
            self._plugin_types[plugin_type].remove(plugin_id)

        # Remove from capability index
        for cap_name, plugin_ids in self._capabilities_index.items():
            if plugin_id in plugin_ids:
                plugin_ids.remove(plugin_id)

        # Remove from language/framework/domain indexes
        for lang_plugins in self._language_index.values():
            if plugin_id in lang_plugins:
                lang_plugins.remove(plugin_id)

        for framework_plugins in self._framework_index.values():
            if plugin_id in framework_plugins:
                framework_plugins.remove(plugin_id)

        for domain_plugins in self._domain_index.values():
            if plugin_id in domain_plugins:
                domain_plugins.remove(plugin_id)

        # Remove plugin
        del self._plugins[plugin_id]

        logger.info(f"Plugin {plugin_id} unregistered")
        return True

    def get_plugin(self, plugin_id: str) -> Optional[BasePlugin]:
        """Get plugin by ID."""
        return self._plugins.get(plugin_id)

    def get_plugins_by_type(self, plugin_type: str) -> List[BasePlugin]:
        """Get all plugins of a specific type."""
        plugin_ids = self._plugin_types.get(plugin_type, [])
        return [self._plugins[pid] for pid in plugin_ids if pid in self._plugins]

    def get_plugins_by_capability(self, capability: str) -> List[BasePlugin]:
        """Get plugins that have a specific capability."""
        plugin_ids = self._capabilities_index.get(capability, [])
        return [self._plugins[pid] for pid in plugin_ids if pid in self._plugins]

    def get_plugins_for_language(self, language: str) -> List[BasePlugin]:
        """Get plugins that support a specific language."""
        plugin_ids = self._language_index.get(language, [])
        return [self._plugins[pid] for pid in plugin_ids if pid in self._plugins]

    def get_plugins_for_framework(self, framework: str) -> List[BasePlugin]:
        """Get plugins that support a specific framework."""
        plugin_ids = self._framework_index.get(framework, [])
        return [self._plugins[pid] for pid in plugin_ids if pid in self._plugins]

    def get_plugins_for_domain(self, domain: str) -> List[BasePlugin]:
        """Get plugins that support a specific domain."""
        plugin_ids = self._domain_index.get(domain, [])
        return [self._plugins[pid] for pid in plugin_ids if pid in self._plugins]

    def find_best_plugin(self, language: str = None, framework: str = None,
                        domain: str = None, capability: str = None) -> Optional[BasePlugin]:
        """Find the best plugin for given criteria."""
        candidates = []

        # Start with all plugins
        candidate_ids = set(self._plugins.keys())

        # Filter by criteria
        if language:
            lang_plugins = set(self._language_index.get(language, []))
            candidate_ids &= lang_plugins

        if framework:
            framework_plugins = set(self._framework_index.get(framework, []))
            candidate_ids &= framework_plugins

        if domain:
            domain_plugins = set(self._domain_index.get(domain, []))
            candidate_ids &= domain_plugins

        if capability:
            cap_plugins = set(self._capabilities_index.get(capability, []))
            candidate_ids &= cap_plugins

        candidates = [self._plugins[pid] for pid in candidate_ids if pid in self._plugins]

        if not candidates:
            return None

        # Select best plugin by priority, then quality score
        candidates.sort(key=lambda p: (p.metadata.priority, p.metadata.quality_score), reverse=True)
        return candidates[0]

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all registered plugins."""
        return [{
            'id': plugin.metadata.plugin_id,
            'name': plugin.metadata.name,
            'version': plugin.metadata.version,
            'type': plugin.metadata.plugin_type,
            'languages': plugin.metadata.supported_languages,
            'frameworks': plugin.metadata.supported_frameworks,
            'domains': plugin.metadata.supported_domains,
            'quality_score': plugin.metadata.quality_score,
            'loaded': plugin.metadata.is_loaded
        } for plugin in self._plugins.values()]

    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            'total_plugins': len(self._plugins),
            'plugins_by_type': {ptype: len(pids) for ptype, pids in self._plugin_types.items()},
            'supported_languages': list(self._language_index.keys()),
            'supported_frameworks': list(self._framework_index.keys()),
            'supported_domains': list(self._domain_index.keys()),
            'capabilities_available': list(self._capabilities_index.keys())
        }


class PluginLoader:
    """Plugin loading and management system."""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self.plugin_directories: List[Path] = []
        self.loaded_plugins: Dict[str, Path] = {}

    def add_plugin_directory(self, directory: Path):
        """Add a directory to search for plugins."""
        if directory.exists() and directory.is_dir():
            self.plugin_directories.append(directory)
            logger.info(f"Added plugin directory: {directory}")

    def discover_plugins(self) -> List[Path]:
        """Discover plugin files in configured directories."""
        plugin_files = []

        for directory in self.plugin_directories:
            # Look for Python files that might be plugins
            for py_file in directory.glob("**/*.py"):
                if self._is_plugin_file(py_file):
                    plugin_files.append(py_file)

            # Look for plugin metadata files
            for json_file in directory.glob("**/plugin.json"):
                plugin_files.append(json_file)

        logger.info(f"Discovered {len(plugin_files)} potential plugin files")
        return plugin_files

    def _is_plugin_file(self, file_path: Path) -> bool:
        """Check if a file is likely a plugin."""
        # Skip common non-plugin files
        if file_path.name.startswith('_') or file_path.name.startswith('.'):
            return False

        # Check if it contains plugin-like classes
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read(2000)  # Read first 2KB

            # Look for plugin base classes
            plugin_indicators = [
                'class.*Plugin(BasePlugin)',
                'class.*Plugin(LanguagePlugin)',
                'class.*Plugin(FrameworkPlugin)',
                'class.*Plugin(DomainPlugin)',
                'class.*Plugin(ToolPlugin)'
            ]

            return any(re.search(pattern, content, re.IGNORECASE) for pattern in plugin_indicators)

        except Exception:
            return False

    def load_plugin(self, plugin_path: Path, config: Dict[str, Any] = None) -> bool:
        """Load a plugin from file."""
        if config is None:
            config = {}

        try:
            plugin_id = self._get_plugin_id(plugin_path)

            if plugin_path.suffix == '.py':
                return self._load_python_plugin(plugin_path, config)
            elif plugin_path.suffix == '.json':
                return self._load_json_plugin(plugin_path, config)
            else:
                logger.error(f"Unsupported plugin file type: {plugin_path}")
                return False

        except Exception as e:
            logger.error(f"Failed to load plugin {plugin_path}: {e}")
            return False

    def _load_python_plugin(self, plugin_path: Path, config: Dict[str, Any]) -> bool:
        """Load a Python plugin."""
        try:
            # Import the module
            module_name = f"plugin_{plugin_path.stem}_{hash(plugin_path)}"
            spec = importlib.util.spec_from_file_location(module_name, plugin_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Find plugin classes
            plugin_classes = []
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and
                    issubclass(obj, BasePlugin) and
                    obj != BasePlugin and
                    obj != LanguagePlugin and
                    obj != FrameworkPlugin and
                    obj != DomainPlugin and
                    obj != ToolPlugin):
                    plugin_classes.append(obj)

            if not plugin_classes:
                logger.error(f"No plugin classes found in {plugin_path}")
                return False

            # Load first plugin class found
            plugin_class = plugin_classes[0]
            plugin_instance = plugin_class()

            # Register plugin
            if self.registry.register_plugin(plugin_instance):
                plugin_instance.metadata.is_loaded = True
                plugin_instance.metadata.load_path = plugin_path
                self.loaded_plugins[plugin_instance.metadata.plugin_id] = plugin_path

                # Initialize plugin
                return plugin_instance.initialize(config)

            return False

        except Exception as e:
            logger.error(f"Python plugin loading failed: {e}")
            return False

    def _load_json_plugin(self, plugin_path: Path, config: Dict[str, Any]) -> bool:
        """Load a JSON plugin configuration."""
        # JSON plugins would define external tool plugins
        # Implementation would create ToolPlugin instances
        logger.info(f"JSON plugin loading not yet implemented: {plugin_path}")
        return False

    def _get_plugin_id(self, plugin_path: Path) -> str:
        """Generate plugin ID from path."""
        path_str = str(plugin_path)
        return hashlib.md5(path_str.encode()).hexdigest()[:8]

    def unload_plugin(self, plugin_id: str) -> bool:
        """Unload a plugin."""
        plugin = self.registry.get_plugin(plugin_id)
        if plugin:
            plugin.cleanup()

        if plugin_id in self.loaded_plugins:
            del self.loaded_plugins[plugin_id]

        return self.registry.unregister_plugin(plugin_id)

    def reload_plugin(self, plugin_id: str, config: Dict[str, Any] = None) -> bool:
        """Reload a plugin."""
        if config is None:
            config = {}

        plugin_path = self.loaded_plugins.get(plugin_id)
        if not plugin_path:
            return False

        # Unload and reload
        self.unload_plugin(plugin_id)
        return self.load_plugin(plugin_path, config)


class PluginOrchestrator:
    """Orchestrate plugin execution for testing tasks."""

    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self.execution_cache: Dict[str, Any] = {}

    def analyze_code_with_plugins(self, code: str, language: str, file_path: Path) -> Dict[str, Any]:
        """Analyze code using available plugins."""
        result = {
            'language': language,
            'file_path': str(file_path),
            'functions': [],
            'complexity': {},
            'framework': None,
            'domain': None,
            'plugin_results': {}
        }

        # Get language plugins
        lang_plugins = self.registry.get_plugins_for_language(language)
        for plugin in lang_plugins:
            if plugin.is_capable('code_analysis'):
                try:
                    analysis = plugin.parse_code(code, file_path)
                    result['plugin_results'][plugin.metadata.plugin_id] = analysis

                    # Merge results
                    if 'functions' in analysis:
                        result['functions'].extend(analysis['functions'])
                    if 'complexity' in analysis:
                        result['complexity'].update(analysis['complexity'])

                except Exception as e:
                    logger.error(f"Plugin {plugin.metadata.plugin_id} analysis failed: {e}")

        # Detect framework
        framework_plugins = self.registry.get_plugins_by_type('framework')
        for plugin in framework_plugins:
            if plugin.is_capable('framework_detection'):
                try:
                    detected = plugin.detect_framework(file_path.parent)
                    if detected:
                        result['framework'] = detected
                        break
                except Exception as e:
                    logger.error(f"Framework detection failed in {plugin.metadata.plugin_id}: {e}")

        # Analyze domain
        domain_plugins = self.registry.get_plugins_by_type('domain')
        for plugin in domain_plugins:
            if plugin.is_capable('domain_intelligence'):
                try:
                    domain = plugin.analyze_domain(code, result['functions'])
                    if domain:
                        result['domain'] = domain
                        break
                except Exception as e:
                    logger.error(f"Domain analysis failed in {plugin.metadata.plugin_id}: {e}")

        return result

    def generate_tests_with_plugins(self, functions: List[Dict[str, Any]],
                                  language: str, framework: str, domain: str = None) -> List[Dict[str, Any]]:
        """Generate tests using plugins."""
        test_candidates = []

        # Language-specific test generation
        lang_plugins = self.registry.get_plugins_for_language(language)
        for plugin in lang_plugins:
            if plugin.is_capable('test_generation'):
                try:
                    tests = plugin.generate_language_specific_tests(functions)
                    test_candidates.extend(tests)
                except Exception as e:
                    logger.error(f"Language test generation failed in {plugin.metadata.plugin_id}: {e}")

        # Framework-specific test generation
        if framework:
            framework_plugins = self.registry.get_plugins_for_framework(framework)
            for plugin in framework_plugins:
                if plugin.is_capable('test_generation'):
                    try:
                        tests = plugin.generate_framework_tests(functions, framework)
                        test_candidates.extend(tests)
                    except Exception as e:
                        logger.error(f"Framework test generation failed in {plugin.metadata.plugin_id}: {e}")

        # Domain-specific test generation
        if domain:
            domain_plugins = self.registry.get_plugins_for_domain(domain)
            for plugin in domain_plugins:
                if plugin.is_capable('test_generation'):
                    try:
                        tests = plugin.generate_domain_tests(functions, domain)
                        test_candidates.extend(tests)
                    except Exception as e:
                        logger.error(f"Domain test generation failed in {plugin.metadata.plugin_id}: {e}")

        return test_candidates

    def validate_with_plugins(self, test_code: str, language: str, framework: str = None,
                            domain: str = None) -> Dict[str, Any]:
        """Validate test code using plugins."""
        validation_results = {
            'overall_score': 1.0,
            'issues': [],
            'plugin_validations': {}
        }

        # Framework validation
        if framework:
            framework_plugins = self.registry.get_plugins_for_framework(framework)
            for plugin in framework_plugins:
                if plugin.is_capable('quality_validation'):
                    try:
                        validation = plugin.validate_framework_usage(test_code, framework)
                        validation_results['plugin_validations'][plugin.metadata.plugin_id] = validation

                        # Adjust overall score
                        if 'score' in validation:
                            validation_results['overall_score'] = min(
                                validation_results['overall_score'], validation['score']
                            )

                        # Collect issues
                        if 'issues' in validation:
                            validation_results['issues'].extend(validation['issues'])

                    except Exception as e:
                        logger.error(f"Framework validation failed in {plugin.metadata.plugin_id}: {e}")

        # Domain validation
        if domain:
            domain_plugins = self.registry.get_plugins_for_domain(domain)
            for plugin in domain_plugins:
                if plugin.is_capable('quality_validation'):
                    try:
                        validation = plugin.validate_domain_correctness(test_code, domain)
                        validation_results['plugin_validations'][plugin.metadata.plugin_id] = validation

                        # Adjust overall score
                        if 'score' in validation:
                            validation_results['overall_score'] = min(
                                validation_results['overall_score'], validation['score']
                            )

                        # Collect issues
                        if 'issues' in validation:
                            validation_results['issues'].extend(validation['issues'])

                    except Exception as e:
                        logger.error(f"Domain validation failed in {plugin.metadata.plugin_id}: {e}")

        return validation_results


# Plugin development utilities
def create_plugin_template(plugin_type: str, plugin_id: str, name: str,
                          supported_languages: List[str] = None,
                          supported_frameworks: List[str] = None,
                          supported_domains: List[str] = None) -> str:
    """Create a plugin template."""
    if supported_languages is None:
        supported_languages = []
    if supported_frameworks is None:
        supported_frameworks = []
    if supported_domains is None:
        supported_domains = []

    base_classes = {
        'language': 'LanguagePlugin',
        'framework': 'FrameworkPlugin',
        'domain': 'DomainPlugin',
        'tool': 'ToolPlugin'
    }

    base_class = base_classes.get(plugin_type, 'BasePlugin')

    template = f'''"""
{plugin_type.capitalize()} Plugin: {name}

Description of what this plugin does.
"""

from unified_test_harness.plugin_system import {base_class}, PluginMetadata, PluginCapabilities

class {plugin_id}Plugin({base_class}):
    """{name} plugin implementation."""

    def __init__(self):
        metadata = PluginMetadata(
            plugin_id="{plugin_id}",
            name="{name}",
            version="1.0.0",
            description="Description of {name} plugin",
            author="Plugin Author",
            license="MIT",
            supported_languages={supported_languages},
            supported_frameworks={supported_frameworks},
            supported_domains={supported_domains},
            plugin_type="{plugin_type}",
            priority=50
        )

        capabilities = PluginCapabilities(
            # Set appropriate capabilities
            code_analysis={plugin_type == "language"},
            test_generation=True,
            quality_validation=True
        )

        super().__init__(metadata, capabilities)

    # Implement required methods based on plugin type
    # See base class documentation for method signatures
'''

    return template


# Example built-in plugins
class PythonLanguagePlugin(LanguagePlugin):
    """Built-in Python language plugin."""

    def __init__(self):
        metadata = PluginMetadata(
            plugin_id="python_lang",
            name="Python Language Support",
            version="1.0.0",
            description="Core Python language analysis and test generation",
            author="VectorReVamp",
            license="MIT",
            supported_languages=["python"],
            plugin_type="language",
            priority=100  # High priority for core language
        )

        capabilities = PluginCapabilities(
            code_analysis=True,
            test_generation=True,
            language_parsing=True
        )

        super().__init__(metadata, capabilities)

    def parse_code(self, code: str, file_path: Path) -> Dict[str, Any]:
        """Parse Python code using AST."""
        import ast
        result = {'functions': [], 'classes': [], 'imports': []}

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = {
                        'name': node.name,
                        'line': node.lineno,
                        'args': [arg.arg for arg in node.args.args],
                        'complexity': self._calculate_complexity(node)
                    }
                    result['functions'].append(func_info)
                elif isinstance(node, ast.ClassDef):
                    result['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    result['imports'].append(ast.get_source_segment(code, node))

        except SyntaxError:
            result['parse_error'] = True

        return result

    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate function complexity."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                complexity += 1

        return complexity

    def extract_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function information."""
        analysis = self.parse_code(code, Path("dummy.py"))
        return analysis.get('functions', [])

    def analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity."""
        functions = self.extract_functions(code)
        total_complexity = sum(f.get('complexity', 1) for f in functions)
        avg_complexity = total_complexity / max(1, len(functions))

        return {
            'total_functions': len(functions),
            'total_complexity': total_complexity,
            'average_complexity': avg_complexity,
            'complexity_distribution': {
                'simple': len([f for f in functions if f.get('complexity', 1) <= 2]),
                'medium': len([f for f in functions if 3 <= f.get('complexity', 1) <= 5]),
                'complex': len([f for f in functions if f.get('complexity', 1) > 5])
            }
        }

    def generate_language_specific_tests(self, functions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate Python-specific tests."""
        tests = []

        for func in functions:
            test = {
                'function_name': func['name'],
                'test_type': 'unit',
                'language': 'python',
                'framework': 'pytest',  # Default assumption
                'test_code': self._generate_pytest_function(func),
                'expected_coverage': 0.8,
                'complexity': func.get('complexity', 1)
            }
            tests.append(test)

        return tests

    def _generate_pytest_function(self, func: Dict[str, Any]) -> str:
        """Generate a pytest function."""
        func_name = func['name']
        args = func.get('args', [])

        test_code = f'''def test_{func_name}():
    """Test {func_name} function"""
    # Setup test data
'''

        if args:
            test_code += f'    # Create test arguments\n'
            for arg in args[:3]:  # Limit to first 3 args
                test_code += f'    {arg} = None  # Test value\n'

        test_code += f'''
    # Call function
    result = {func_name}({', '.join(args[:3])})

    # Assertions
    assert result is not None
'''

        return test_code
