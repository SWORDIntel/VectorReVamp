"""
Coverage Analysis and Gap Detection

Analyzes code coverage and identifies gaps for test generation.
Framework-agnostic implementation.
Supports Python (pytest-cov), C (gcov/lcov), and Rust (cargo-tarpaulin).
"""

import ast
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import xml.etree.ElementTree as ET
import re

from .language_parser import LanguageParser, Language

# Configure logging
logger = logging.getLogger(__name__)

# Default timeout for subprocess calls (30 seconds)
DEFAULT_TIMEOUT = 30


class CoverageAnalyzer:
    """Analyzes code coverage and identifies gaps"""
    
    def __init__(self, config):
        """
        Initialize coverage analyzer
        
        Args:
            config: HarnessConfig instance
        """
        self.config = config
        self.source_root = config.source_root
        self.module_functions: Dict[str, List[str]] = {}
        self.module_classes: Dict[str, List[str]] = {}
        self.coverage_data: Dict[str, float] = {}
        self.coverage_xml: Optional[Path] = None
        self.language_parser = LanguageParser()
    
    def analyze_module_structure(self, module_path: Path) -> Dict[str, List[str]]:
        """Analyze module structure to find functions and classes"""
        language = self.language_parser.detect_language(module_path)
        
        if language == Language.PYTHON:
            return self._analyze_python_structure(module_path)
        elif language == Language.C:
            return self._analyze_c_structure(module_path)
        elif language == Language.RUST:
            return self._analyze_rust_structure(module_path)
        elif language == Language.CPP:
            return self._analyze_cpp_structure(module_path)
        
        return {'functions': [], 'classes': [], 'structs': []}
    
    def _analyze_python_structure(self, module_path: Path) -> Dict[str, List[str]]:
        """Analyze Python module structure"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(module_path))
        except (SyntaxError, IndentationError, UnicodeDecodeError) as e:
            return {'functions': [], 'classes': []}
        
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_') or node.name.startswith('__'):
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = item.name
                        if not method_name.startswith('_') or method_name.startswith('__'):
                            functions.append(f"{node.name}.{method_name}")
        
        return {'functions': functions, 'classes': classes}
    
    def _analyze_c_structure(self, module_path: Path) -> Dict[str, List[str]]:
        """Analyze C module structure"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return {'functions': [], 'structs': []}
        
        functions = []
        structs = []
        
        # Find functions
        function_pattern = re.compile(r'^(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{', re.MULTILINE)
        for match in function_pattern.finditer(content):
            func_name = match.group(1)
            if not func_name.startswith('_') or func_name.startswith('__'):
                functions.append(func_name)
        
        # Find structs
        struct_pattern = re.compile(r'struct\s+(\w+)\s*\{', re.MULTILINE)
        for match in struct_pattern.finditer(content):
            structs.append(match.group(1))
        
        return {'functions': functions, 'structs': structs}
    
    def _analyze_rust_structure(self, module_path: Path) -> Dict[str, List[str]]:
        """Analyze Rust module structure"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return {'functions': [], 'structs': [], 'traits': []}
        
        functions = []
        structs = []
        traits = []
        
        # Find functions
        function_pattern = re.compile(r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(', re.MULTILINE)
        for match in function_pattern.finditer(content):
            func_name = match.group(1)
            if not func_name.startswith('_'):
                functions.append(func_name)
        
        # Find structs
        struct_pattern = re.compile(r'^\s*(?:pub\s+)?struct\s+(\w+)\s*\{', re.MULTILINE)
        for match in struct_pattern.finditer(content):
            structs.append(match.group(1))
        
        # Find traits
        trait_pattern = re.compile(r'^\s*(?:pub\s+)?trait\s+(\w+)\s*\{', re.MULTILINE)
        for match in trait_pattern.finditer(content):
            traits.append(match.group(1))
        
        return {'functions': functions, 'structs': structs, 'traits': traits}
    
    def _analyze_cpp_structure(self, module_path: Path) -> Dict[str, List[str]]:
        """Analyze C++ module structure"""
        c_structure = self._analyze_c_structure(module_path)
        
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return c_structure
        
        classes = []
        class_pattern = re.compile(r'class\s+(\w+)\s*\{', re.MULTILINE)
        for match in class_pattern.finditer(content):
            classes.append(match.group(1))
        
        c_structure['classes'] = classes
        return c_structure
    
    def parse_coverage_xml(self, xml_path: Path) -> Dict[str, float]:
        """Parse coverage XML to get coverage percentages"""
        if not xml_path.exists():
            return {}
        
        coverage = {}
        
        # Try to detect XML format (Cobertura, LCOV, tarpaulin, etc.)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Cobertura format (Python pytest-cov, Rust tarpaulin)
            if root.tag == 'coverage' or root.find('.//package') is not None:
                for package in root.findall('.//package'):
                    for class_elem in package.findall('.//class'):
                        class_name = class_elem.get('name', '')
                        for method in class_elem.findall('.//method'):
                            method_name = method.get('name', '')
                            line_rate = float(method.get('line-rate', 0))
                            full_name = f"{class_name}.{method_name}" if class_name else method_name
                            coverage[full_name] = line_rate
                        
                        # File-level coverage
                        filename = class_elem.get('filename', '')
                        line_rate = float(class_elem.get('line-rate', 0))
                        if filename:
                            coverage[filename] = line_rate
            
            # Tarpaulin format (Rust)
            elif root.tag == 'coverage':
                for package in root.findall('.//package'):
                    name = package.get('name', '')
                    line_rate = float(package.get('line-rate', 0))
                    coverage[name] = line_rate
            
        except Exception as e:
            logger.warning(f"Error parsing coverage XML: {e}")
        
        return coverage
    
    def parse_lcov(self, lcov_path: Path) -> Dict[str, float]:
        """Parse LCOV format (used by gcov for C)"""
        coverage = {}
        
        if not lcov_path.exists():
            return coverage
        
        try:
            with open(lcov_path, 'r') as f:
                content = f.read()
            
            # Parse LCOV format
            current_file = None
            lines_covered = 0
            lines_total = 0
            
            for line in content.splitlines():
                if line.startswith('SF:'):
                    # Source file
                    current_file = line[3:].strip()
                    lines_covered = 0
                    lines_total = 0
                elif line.startswith('DA:'):
                    # Line data: DA:line_number,execution_count
                    parts = line[3:].split(',')
                    if len(parts) == 2:
                        lines_total += 1
                        if int(parts[1]) > 0:
                            lines_covered += 1
                elif line == 'end_of_record' and current_file:
                    # End of file record
                    if lines_total > 0:
                        coverage[current_file] = lines_covered / lines_total
                    current_file = None
        
        except Exception as e:
            logger.warning(f"Error parsing LCOV file: {e}")
        
        return coverage
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis using configured test framework"""
        logger.info("Running coverage analysis...")
        
        # Detect language from source files
        source_patterns = self.config.framework.source_patterns
        languages = set()
        for pattern in source_patterns:
            for src_file in self.source_root.glob(pattern):
                lang = self.language_parser.detect_language(src_file)
                if lang:
                    languages.add(lang)
        
        # Determine coverage tool based on detected languages
        if Language.PYTHON in languages:
            return self._run_python_coverage()
        elif Language.C in languages or Language.CPP in languages:
            return self._run_c_coverage()
        elif Language.RUST in languages:
            return self._run_rust_coverage()
        else:
            # Default to Python
            return self._run_python_coverage()
    
    def _run_python_coverage(self) -> Dict[str, Any]:
        """Run Python coverage analysis"""
        coverage_xml = self.config.output_dir / "coverage.xml"
        coverage_html = self.config.output_dir / "htmlcov"
        
        cmd = self.config.framework.coverage_command.copy()
        
        if "--cov-report=xml" in cmd:
            idx = cmd.index("--cov-report=xml")
            cmd[idx] = f"--cov-report=xml:{coverage_xml}"
        else:
            cmd.extend(["--cov-report=xml", str(coverage_xml)])
        
        if "--cov-report=html" not in cmd:
            cmd.extend(["--cov-report=html", str(coverage_html)])
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.source_root,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT * 20  # 10 minutes for Python coverage
            )
            
            if result.returncode != 0:
                logger.error(f"Coverage command failed: {result.stderr}")
                return {}
            
            if coverage_xml.exists():
                self.coverage_xml = coverage_xml
                self.coverage_data = self.parse_coverage_xml(coverage_xml)
                return self.get_coverage_report()
            else:
                logger.warning("Coverage XML not generated")
                return {}
                
        except subprocess.TimeoutExpired:
            logger.error("Coverage command timed out")
            return {}
        except Exception as e:
            logger.error(f"Error running Python coverage: {e}", exc_info=True)
            return {}
    
    def _run_c_coverage(self) -> Dict[str, Any]:
        """Run C coverage analysis using gcov/lcov"""
        logger.info("Running C coverage with gcov/lcov...")
        
        coverage_dir = self.config.output_dir / "coverage"
        coverage_dir.mkdir(exist_ok=True)
        
        lcov_file = self.config.output_dir / "coverage.info"
        coverage_xml = self.config.output_dir / "coverage.xml"
        
        # Try to run tests with coverage
        # This assumes the project has been built with --coverage flag
        try:
            # Run lcov to generate coverage.info
            result = subprocess.run(
                ["lcov", "--capture", "--directory", str(self.source_root), 
                 "--output-file", str(lcov_file)],
                cwd=self.source_root,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT * 20  # 10 minutes for C coverage
            )
            
            if result.returncode == 0 and lcov_file.exists():
                self.coverage_data = self.parse_lcov(lcov_file)
                # Convert to XML-like format for compatibility
                self.coverage_xml = lcov_file
                return self.get_coverage_report()
            else:
                logger.warning("LCOV generation failed, trying gcov directly...")
                # Try gcov directly
                gcov_files = list(self.source_root.glob("**/*.gcov"))
                if gcov_files:
                    # Parse gcov files
                    return self.get_coverage_report()
                return {}
                
        except FileNotFoundError:
            logger.warning("lcov/gcov not found. Install with: sudo apt-get install lcov")
            return {}
        except subprocess.TimeoutExpired:
            logger.error("C coverage command timed out")
            return {}
        except Exception as e:
            logger.error(f"Error running C coverage: {e}", exc_info=True)
            return {}
    
    def _run_rust_coverage(self) -> Dict[str, Any]:
        """Run Rust coverage analysis using cargo-tarpaulin"""
        logger.info("Running Rust coverage with cargo-tarpaulin...")
        
        coverage_xml = self.config.output_dir / "coverage.xml"
        
        try:
            # Check if cargo-tarpaulin is installed
            result = subprocess.run(
                ["cargo", "tarpaulin", "--version"],
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT
            )
            
            if result.returncode != 0:
                logger.warning("cargo-tarpaulin not found. Install with: cargo install cargo-tarpaulin")
                return {}
            
            # Run tarpaulin
            result = subprocess.run(
                ["cargo", "tarpaulin", "--out", "Xml", "--output-dir", str(self.config.output_dir)],
                cwd=self.source_root,
                capture_output=True,
                text=True,
                timeout=DEFAULT_TIMEOUT * 20  # 10 minutes for Rust coverage
            )
            
            # Tarpaulin outputs to cobertura.xml
            tarpaulin_xml = self.config.output_dir / "cobertura.xml"
            if not tarpaulin_xml.exists():
                tarpaulin_xml = self.config.output_dir / "tarpaulin-report.xml"
            
            if tarpaulin_xml.exists():
                # Copy to coverage.xml for consistency
                import shutil
                shutil.copy(tarpaulin_xml, coverage_xml)
                self.coverage_xml = coverage_xml
                self.coverage_data = self.parse_coverage_xml(coverage_xml)
                return self.get_coverage_report()
            else:
                logger.warning("Tarpaulin XML not generated")
                return {}
                
        except FileNotFoundError:
            logger.warning("cargo not found. Make sure Rust is installed.")
            return {}
        except subprocess.TimeoutExpired:
            logger.error("Rust coverage command timed out")
            return {}
        except Exception as e:
            logger.error(f"Error running Rust coverage: {e}", exc_info=True)
            return {}
    
    def get_uncovered_functions(self, module_name: str) -> List[str]:
        """Get list of uncovered functions for a module"""
        # Find module file
        module_path = self._find_module_file(module_name)
        if not module_path or not module_path.exists():
            return []
        
        structure = self.analyze_module_structure(module_path)
        all_functions = structure['functions']
        
        # Get coverage data
        if self.coverage_xml:
            coverage = self.parse_coverage_xml(self.coverage_xml)
        else:
            coverage = {}
        
        uncovered = []
        for func in all_functions:
            # Check if function is covered
            func_coverage = coverage.get(func, 0.0)
            if func_coverage < self.config.coverage_threshold:
                uncovered.append(func)
        
        return uncovered
    
    def _find_module_file(self, module_name: str) -> Optional[Path]:
        """Find module file by name (supports Python, C, Rust)"""
        # Try different patterns and extensions
        extensions = ['.py', '.c', '.h', '.rs', '.cpp', '.cc', '.cxx']
        
        for ext in extensions:
            patterns = [
                f"{module_name}{ext}",
                f"**/{module_name}{ext}",
            ]
            
            for pattern in patterns:
                matches = list(self.source_root.glob(pattern))
                if matches:
                    return matches[0]
        
        # Try with import prefix
        if self.config.framework.import_prefix:
            prefix = self.config.framework.import_prefix.rstrip('.')
            for ext in extensions:
                patterns = [
                    f"{prefix}/{module_name}{ext}",
                    f"{prefix}/**/{module_name}{ext}",
                ]
                for pattern in patterns:
                    matches = list(self.source_root.glob(pattern))
                    if matches:
                        return matches[0]
        
        return None
    
    def identify_zero_coverage_modules(self, threshold: float = 0.01) -> List[Dict[str, Any]]:
        """Identify modules with zero or very low coverage"""
        zero_coverage = []
        
        # Find all Python modules
        source_patterns = self.config.framework.source_patterns
        all_modules = set()
        
        for pattern in source_patterns:
            for file_path in self.source_root.glob(pattern):
                if file_path.name == "__init__.py" or "__pycache__" in str(file_path):
                    continue
                if file_path.suffix in [".py", ".c", ".h", ".rs", ".cpp", ".cc", ".cxx"]:
                    module_name = file_path.stem
                    all_modules.add((module_name, file_path))
        
        for module_name, module_file in all_modules:
            coverage_pct = self.coverage_data.get(module_name, 0.0)
            
            if coverage_pct <= threshold:
                structure = self.analyze_module_structure(module_file)
                zero_coverage.append({
                    'module': module_name,
                    'file_path': str(module_file.relative_to(self.source_root)),
                    'coverage': coverage_pct,
                    'functions': len(structure['functions']),
                    'classes': len(structure['classes']),
                    'priority': 'high' if coverage_pct == 0.0 else 'medium'
                })
        
        return sorted(zero_coverage, key=lambda x: (x['coverage'], -x['functions']))
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Generate comprehensive coverage report"""
        report = {
            'modules': {},
            'total_functions': 0,
            'covered_functions': 0,
            'coverage_percentage': 0.0,
            'gaps': []
        }
        
        if self.coverage_xml:
            coverage = self.parse_coverage_xml(self.coverage_xml)
        else:
            coverage = {}
        
        # Find all modules
        source_patterns = self.config.framework.source_patterns
        all_modules = set()
        
        for pattern in source_patterns:
            for file_path in self.source_root.glob(pattern):
                if file_path.name == "__init__.py" or "__pycache__" in str(file_path):
                    continue
                if file_path.suffix in [".py", ".c", ".h", ".rs", ".cpp", ".cc", ".cxx"]:
                    module_name = file_path.stem
                    all_modules.add((module_name, file_path))
        
        for module_name, module_file in all_modules:
            structure = self.analyze_module_structure(module_file)
            
            module_report = {
                'file_path': str(module_file.relative_to(self.source_root)),
                'functions': len(structure['functions']),
                'classes': len(structure['classes']),
                'covered_functions': 0,
                'coverage_percentage': 0.0,
                'uncovered': []
            }
            
            for func in structure['functions']:
                func_coverage = coverage.get(func, 0.0)
                if func_coverage >= self.config.coverage_threshold:
                    module_report['covered_functions'] += 1
                else:
                    module_report['uncovered'].append(func)
            
            if module_report['functions'] > 0:
                module_report['coverage_percentage'] = (
                    module_report['covered_functions'] / module_report['functions'] * 100
                )
            
            report['modules'][module_name] = module_report
            report['total_functions'] += module_report['functions']
            report['covered_functions'] += module_report['covered_functions']
        
        if report['total_functions'] > 0:
            report['coverage_percentage'] = (
                report['covered_functions'] / report['total_functions'] * 100
            )
        
        return report
    
    def identify_test_gaps(self, existing_tests: List[str] = None) -> List[Dict[str, Any]]:
        """Identify gaps in test coverage that need new test vectors"""
        gaps = []
        
        # Find all modules
        source_patterns = self.config.framework.source_patterns
        all_modules = set()
        
        for pattern in source_patterns:
            for file_path in self.source_root.glob(pattern):
                if file_path.name == "__init__.py" or "__pycache__" in str(file_path):
                    continue
                if file_path.suffix in [".py", ".c", ".h", ".rs", ".cpp", ".cc", ".cxx"]:
                    module_name = file_path.stem
                    all_modules.add((module_name, file_path))
        
        for module_name, module_file in all_modules:
            uncovered = self.get_uncovered_functions(module_name)
            
            if uncovered:
                gaps.append({
                    'module': module_name,
                    'file_path': str(module_file.relative_to(self.source_root)),
                    'uncovered_functions': uncovered,
                    'priority': 'high' if len(uncovered) > 5 else 'medium'
                })
        
        return gaps
