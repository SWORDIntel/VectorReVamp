"""
Code Analyzer for Enhanced Test Generation

Extracts detailed information about functions, classes, and modules to improve
LLM test generation quality.
"""

import ast
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from .language_parser import LanguageParser, Language, CodeElement


@dataclass
class FunctionInfo:
    """Detailed information about a function"""
    name: str
    signature: str
    parameters: List[Dict[str, Any]] = field(default_factory=list)  # [{"name": "x", "type": "int", "default": None}]
    return_type: Optional[str] = None
    docstring: Optional[str] = None
    code: str = ""
    start_line: int = 0
    end_line: int = 0
    parent_class: Optional[str] = None
    raises: List[str] = field(default_factory=list)  # Exception types that might be raised
    dependencies: List[str] = field(default_factory=list)  # Functions/classes used
    is_async: bool = False
    is_generator: bool = False


@dataclass
class ModuleInfo:
    """Information about a module"""
    name: str
    file_path: str
    functions: List[FunctionInfo] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    imports: List[str] = field(default_factory=list)
    language: Language = Language.PYTHON


class CodeAnalyzer:
    """Analyzes code to extract detailed information for test generation"""
    
    def __init__(self):
        self.language_parser = LanguageParser()
    
    def analyze_function(self, file_path: Path, function_name: str, 
                        parent_class: Optional[str] = None) -> Optional[FunctionInfo]:
        """Analyze a specific function and extract detailed information"""
        language = self.language_parser.detect_language(file_path)
        
        if language == Language.PYTHON:
            return self._analyze_python_function(file_path, function_name, parent_class)
        elif language == Language.C:
            return self._analyze_c_function(file_path, function_name)
        elif language == Language.RUST:
            return self._analyze_rust_function(file_path, function_name)
        
        return None
    
    def _analyze_python_function(self, file_path: Path, function_name: str,
                                  parent_class: Optional[str] = None) -> Optional[FunctionInfo]:
        """Analyze Python function"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    # Check if it's in the right class
                    if parent_class:
                        # Find parent class
                        for class_node in ast.walk(tree):
                            if isinstance(class_node, ast.ClassDef) and class_node.name == parent_class:
                                if node not in class_node.body:
                                    continue
                    elif parent_class is None:
                        # Make sure it's not in a class
                        for class_node in ast.walk(tree):
                            if isinstance(class_node, ast.ClassDef):
                                if node in class_node.body:
                                    continue
                    
                    return self._extract_python_function_info(node, content, file_path)
        except Exception as e:
            print(f"Error analyzing Python function {function_name}: {e}")
        
        return None
    
    def _extract_python_function_info(self, node: ast.FunctionDef, 
                                      content: str, file_path: Path) -> FunctionInfo:
        """Extract detailed information from Python function AST node"""
        # Extract parameters
        parameters = []
        for arg in node.args.args:
            param_info = {"name": arg.arg}
            
            # Try to get type hint
            if arg.annotation:
                param_info["type"] = ast.unparse(arg.annotation) if hasattr(ast, 'unparse') else str(arg.annotation)
            
            # Check for default value
            defaults_start = len(node.args.args) - len(node.args.defaults)
            if arg in node.args.args[defaults_start:]:
                idx = node.args.args[defaults_start:].index(arg)
                default = node.args.defaults[idx]
                param_info["default"] = ast.unparse(default) if hasattr(ast, 'unparse') else repr(default)
            
            parameters.append(param_info)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        # Extract code
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        code = '\n'.join(lines[start_line-1:end_line])
        
        # Extract exceptions that might be raised
        raises = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if child.exc:
                    exc_name = ast.unparse(child.exc) if hasattr(ast, 'unparse') else str(child.exc)
                    raises.append(exc_name)
        
        # Extract dependencies (function calls)
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(ast.unparse(child.func) if hasattr(ast, 'unparse') else str(child.func))
        
        # Check if async
        is_async = isinstance(node, ast.AsyncFunctionDef)
        
        # Check if generator
        is_generator = any(isinstance(n, ast.Yield) or isinstance(n, ast.YieldFrom) 
                          for n in ast.walk(node))
        
        # Get signature
        signature = ast.unparse(node.args) if hasattr(ast, 'unparse') else str(node.args)
        
        # Get parent class
        parent_class = None
        for parent in ast.walk(ast.parse(content)):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    parent_class = parent.name
                    break
        
        return FunctionInfo(
            name=node.name,
            signature=signature,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            code=code,
            start_line=start_line,
            end_line=end_line,
            parent_class=parent_class,
            raises=raises,
            dependencies=list(set(dependencies)),
            is_async=is_async,
            is_generator=is_generator
        )
    
    def _analyze_c_function(self, file_path: Path, function_name: str) -> Optional[FunctionInfo]:
        """Analyze C function"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find function definition
            pattern = re.compile(
                r'((?:\w+\s+)*)\s*(\w+)\s*\(([^)]*)\)\s*\{',
                re.MULTILINE
            )
            
            for match in pattern.finditer(content):
                if match.group(2) == function_name:
                    return_type = match.group(1).strip()
                    params_str = match.group(3).strip()
                    
                    # Parse parameters
                    parameters = []
                    if params_str and params_str != 'void':
                        for param in params_str.split(','):
                            param = param.strip()
                            # Try to extract name and type
                            parts = param.split()
                            if len(parts) >= 2:
                                param_type = ' '.join(parts[:-1])
                                param_name = parts[-1]
                                parameters.append({
                                    "name": param_name,
                                    "type": param_type
                                })
                    
                    # Extract code
                    start_pos = match.start()
                    start_line = content[:start_pos].count('\n') + 1
                    
                    # Find matching brace
                    brace_count = 0
                    end_pos = start_pos
                    for i, char in enumerate(content[start_pos:], start_pos):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                end_pos = i + 1
                                break
                    
                    end_line = content[:end_pos].count('\n') + 1
                    code = content[start_pos:end_pos]
                    
                    return FunctionInfo(
                        name=function_name,
                        signature=f"{return_type} {function_name}({params_str})",
                        parameters=parameters,
                        return_type=return_type or "void",
                        code=code,
                        start_line=start_line,
                        end_line=end_line
                    )
        except Exception as e:
            print(f"Error analyzing C function {function_name}: {e}")
        
        return None
    
    def _analyze_rust_function(self, file_path: Path, function_name: str) -> Optional[FunctionInfo]:
        """Analyze Rust function"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find function definition
            pattern = re.compile(
                r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\(([^)]*)\)(?:\s*->\s*([^{]+))?\s*\{',
                re.MULTILINE
            )
            
            for match in pattern.finditer(content):
                if match.group(1) == function_name:
                    params_str = match.group(2).strip()
                    return_type = match.group(3).strip() if match.group(3) else None
                    
                    # Parse parameters
                    parameters = []
                    if params_str:
                        for param in params_str.split(','):
                            param = param.strip()
                            if ':' in param:
                                param_name, param_type = param.split(':', 1)
                                parameters.append({
                                    "name": param_name.strip(),
                                    "type": param_type.strip()
                                })
                    
                    # Extract code
                    start_pos = match.start()
                    start_line = content[:start_pos].count('\n') + 1
                    
                    # Find matching brace
                    brace_count = 0
                    end_pos = start_pos
                    in_string = False
                    string_char = None
                    
                    for i, char in enumerate(content[start_pos:], start_pos):
                        if char in ['"', "'"] and (i == start_pos or content[i-1] != '\\'):
                            if not in_string:
                                in_string = True
                                string_char = char
                            elif char == string_char:
                                in_string = False
                                string_char = None
                        
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    end_pos = i + 1
                                    break
                    
                    end_line = content[:end_pos].count('\n') + 1
                    code = content[start_pos:end_pos]
                    
                    return FunctionInfo(
                        name=function_name,
                        signature=f"fn {function_name}({params_str})" + (f" -> {return_type}" if return_type else ""),
                        parameters=parameters,
                        return_type=return_type,
                        code=code,
                        start_line=start_line,
                        end_line=end_line
                    )
        except Exception as e:
            print(f"Error analyzing Rust function {function_name}: {e}")
        
        return None
    
    def analyze_module(self, file_path: Path) -> ModuleInfo:
        """Analyze entire module"""
        language = self.language_parser.detect_language(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            return ModuleInfo(name=file_path.stem, file_path=str(file_path), language=language)
        
        elements = self.language_parser.parse_file(file_path)
        
        functions = []
        classes = []
        imports = []
        
        if language == Language.PYTHON:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
        
        for element in elements:
            if element.type == 'function':
                func_info = self.analyze_function(file_path, element.name, element.parent)
                if func_info:
                    functions.append(func_info)
            elif element.type in ['class', 'struct']:
                classes.append(element.name)
        
        return ModuleInfo(
            name=file_path.stem,
            file_path=str(file_path),
            functions=functions,
            classes=classes,
            imports=imports,
            language=language
        )
