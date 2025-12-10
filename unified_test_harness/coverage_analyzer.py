"""
Coverage Analysis and Gap Detection

Analyzes code coverage and identifies gaps for test generation.
Framework-agnostic implementation.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import json
import subprocess
import xml.etree.ElementTree as ET


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
    
    def analyze_module_structure(self, module_path: Path) -> Dict[str, List[str]]:
        """Analyze module structure to find functions and classes"""
        try:
            with open(module_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(module_path))
        except (SyntaxError, IndentationError, UnicodeDecodeError) as e:
            # Skip files with syntax errors
            return {'functions': [], 'classes': []}
        
        functions = []
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Skip private functions if needed
                if not node.name.startswith('_') or node.name.startswith('__'):
                    functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)
                # Also get methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_name = item.name
                        if not method_name.startswith('_') or method_name.startswith('__'):
                            functions.append(f"{node.name}.{method_name}")
        
        return {
            'functions': functions,
            'classes': classes
        }
    
    def parse_coverage_xml(self, xml_path: Path) -> Dict[str, float]:
        """Parse coverage XML to get coverage percentages"""
        if not xml_path.exists():
            return {}
        
        coverage = {}
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for package in root.findall('.//package'):
            for class_elem in package.findall('.//class'):
                class_name = class_elem.get('name', '')
                for method in class_elem.findall('.//method'):
                    method_name = method.get('name', '')
                    line_rate = float(method.get('line-rate', 0))
                    full_name = f"{class_name}.{method_name}" if class_name else method_name
                    coverage[full_name] = line_rate
        
        # Also get file-level coverage
        for package in root.findall('.//package'):
            for class_elem in package.findall('.//class'):
                filename = class_elem.get('filename', '')
                line_rate = float(class_elem.get('line-rate', 0))
                if filename:
                    coverage[filename] = line_rate
        
        return coverage
    
    def run_coverage_analysis(self) -> Dict[str, Any]:
        """Run coverage analysis using configured test framework"""
        print("[*] Running coverage analysis...")
        
        coverage_xml = self.config.output_dir / "coverage.xml"
        coverage_html = self.config.output_dir / "htmlcov"
        
        # Build coverage command
        cmd = self.config.framework.coverage_command.copy()
        
        # Add output paths
        if "--cov-report=xml" in cmd:
            idx = cmd.index("--cov-report=xml")
            cmd[idx] = f"--cov-report=xml:{coverage_xml}"
        else:
            cmd.extend(["--cov-report=xml", str(coverage_xml)])
        
        if "--cov-report=html" not in cmd:
            cmd.extend(["--cov-report=html", str(coverage_html)])
        
        # Run coverage
        try:
            result = subprocess.run(
                cmd,
                cwd=self.source_root,
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if result.returncode != 0:
                print(f"[!] Coverage command failed: {result.stderr}")
                return {}
            
            # Parse coverage XML
            if coverage_xml.exists():
                self.coverage_xml = coverage_xml
                self.coverage_data = self.parse_coverage_xml(coverage_xml)
                return self.get_coverage_report()
            else:
                print("[!] Coverage XML not generated")
                return {}
                
        except subprocess.TimeoutExpired:
            print("[!] Coverage analysis timed out")
            return {}
        except Exception as e:
            print(f"[!] Error running coverage: {e}")
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
        """Find module file by name"""
        # Try different patterns based on framework config
        patterns = [
            f"{module_name}.py",
            f"**/{module_name}.py",
        ]
        
        for pattern in patterns:
            matches = list(self.source_root.glob(pattern))
            if matches:
                return matches[0]
        
        # Try with import prefix
        if self.config.framework.import_prefix:
            prefix = self.config.framework.import_prefix.rstrip('.')
            patterns = [
                f"{prefix}/{module_name}.py",
                f"{prefix}/**/{module_name}.py",
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
                if file_path.suffix == ".py":
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
                if file_path.suffix == ".py":
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
                if file_path.suffix == ".py":
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
