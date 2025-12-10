"""
Language Parsers for Multiple Languages

Supports parsing Python, C, and Rust code to extract functions, classes, and modules.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Supported languages"""
    PYTHON = "python"
    C = "c"
    RUST = "rust"
    CPP = "cpp"  # C++ support


@dataclass
class CodeElement:
    """Represents a code element (function, struct, class, etc.)"""
    name: str
    type: str  # 'function', 'struct', 'class', 'module', etc.
    file_path: str
    start_line: int
    end_line: int
    signature: Optional[str] = None
    parent: Optional[str] = None  # Parent class/struct/module
    docstring: Optional[str] = None


class LanguageParser:
    """Base class for language parsers"""
    
    def detect_language(self, file_path: Path) -> Optional[Language]:
        """Detect language from file extension"""
        ext = file_path.suffix.lower()
        if ext == '.py':
            return Language.PYTHON
        elif ext in ['.c', '.h']:
            return Language.C
        elif ext == '.rs':
            return Language.RUST
        elif ext in ['.cpp', '.cc', '.cxx', '.hpp', '.hxx']:
            return Language.CPP
        return None
    
    def parse_file(self, file_path: Path) -> List[CodeElement]:
        """Parse a file and return code elements"""
        language = self.detect_language(file_path)
        if not language:
            return []
        
        if language == Language.PYTHON:
            return self._parse_python(file_path)
        elif language == Language.C:
            return self._parse_c(file_path)
        elif language == Language.RUST:
            return self._parse_rust(file_path)
        elif language == Language.CPP:
            return self._parse_cpp(file_path)
        
        return []
    
    def _parse_python(self, file_path: Path) -> List[CodeElement]:
        """Parse Python file using AST"""
        import ast
        elements = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=str(file_path))
            lines = content.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    # Get parent class if exists
                    parent = None
                    for parent_node in ast.walk(tree):
                        if isinstance(parent_node, ast.ClassDef):
                            for item in parent_node.body:
                                if item == node:
                                    parent = parent_node.name
                                    break
                    
                    signature = ast.unparse(node.args) if hasattr(ast, 'unparse') else None
                    
                    elements.append(CodeElement(
                        name=node.name,
                        type='function',
                        file_path=str(file_path),
                        start_line=start_line,
                        end_line=end_line,
                        signature=signature,
                        parent=parent,
                        docstring=ast.get_docstring(node)
                    ))
                elif isinstance(node, ast.ClassDef):
                    start_line = node.lineno
                    end_line = node.end_lineno or start_line
                    
                    elements.append(CodeElement(
                        name=node.name,
                        type='class',
                        file_path=str(file_path),
                        start_line=start_line,
                        end_line=end_line,
                        docstring=ast.get_docstring(node)
                    ))
        except Exception as e:
            print(f"Error parsing Python file {file_path}: {e}")
        
        return elements
    
    def _parse_c(self, file_path: Path) -> List[CodeElement]:
        """Parse C file using regex patterns"""
        elements = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            # Pattern for function definitions
            # Matches: return_type function_name(args) { ... }
            function_pattern = re.compile(
                r'^(?:\w+\s+)*(\w+)\s*\([^)]*\)\s*\{',
                re.MULTILINE
            )
            
            # Pattern for struct definitions
            struct_pattern = re.compile(
                r'struct\s+(\w+)\s*\{',
                re.MULTILINE
            )
            
            # Find functions
            for match in function_pattern.finditer(content):
                func_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
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
                
                # Extract signature
                func_start = content.rfind('\n', 0, start_pos) + 1
                signature = content[func_start:start_pos].strip()
                
                elements.append(CodeElement(
                    name=func_name,
                    type='function',
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature
                ))
            
            # Find structs
            for match in struct_pattern.finditer(content):
                struct_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
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
                
                elements.append(CodeElement(
                    name=struct_name,
                    type='struct',
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line
                ))
        
        except Exception as e:
            print(f"Error parsing C file {file_path}: {e}")
        
        return elements
    
    def _parse_rust(self, file_path: Path) -> List[CodeElement]:
        """Parse Rust file using regex patterns"""
        elements = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            lines = content.splitlines()
            
            # Pattern for function definitions: fn function_name(args) -> return_type { ... }
            function_pattern = re.compile(
                r'^\s*(?:pub\s+)?(?:async\s+)?fn\s+(\w+)\s*\([^)]*\)(?:\s*->\s*[^{]+)?\s*\{',
                re.MULTILINE
            )
            
            # Pattern for struct definitions: struct StructName { ... }
            struct_pattern = re.compile(
                r'^\s*(?:pub\s+)?struct\s+(\w+)\s*\{',
                re.MULTILINE
            )
            
            # Pattern for impl blocks: impl StructName { ... }
            impl_pattern = re.compile(
                r'^\s*(?:pub\s+)?impl\s+(?:\w+::)*(\w+)\s*\{',
                re.MULTILINE
            )
            
            # Pattern for trait definitions: trait TraitName { ... }
            trait_pattern = re.compile(
                r'^\s*(?:pub\s+)?trait\s+(\w+)\s*\{',
                re.MULTILINE
            )
            
            # Find functions
            for match in function_pattern.finditer(content):
                func_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
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
                
                # Extract signature
                func_start = content.rfind('\n', 0, start_pos) + 1
                signature = content[func_start:start_pos + match.end() - match.start()].strip()
                
                elements.append(CodeElement(
                    name=func_name,
                    type='function',
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    signature=signature
                ))
            
            # Find structs
            for match in struct_pattern.finditer(content):
                struct_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
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
                
                elements.append(CodeElement(
                    name=struct_name,
                    type='struct',
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line
                ))
            
            # Find impl blocks
            for match in impl_pattern.finditer(content):
                impl_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
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
                
                elements.append(CodeElement(
                    name=impl_name,
                    type='impl',
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line
                ))
            
            # Find traits
            for match in trait_pattern.finditer(content):
                trait_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
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
                
                elements.append(CodeElement(
                    name=trait_name,
                    type='trait',
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line
                ))
        
        except Exception as e:
            print(f"Error parsing Rust file {file_path}: {e}")
        
        return elements
    
    def _parse_cpp(self, file_path: Path) -> List[CodeElement]:
        """Parse C++ file - similar to C but with class support"""
        elements = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use C parser for functions and structs
            c_elements = self._parse_c(file_path)
            elements.extend(c_elements)
            
            # Pattern for class definitions
            class_pattern = re.compile(
                r'class\s+(\w+)\s*\{',
                re.MULTILINE
            )
            
            # Find classes
            for match in class_pattern.finditer(content):
                class_name = match.group(1)
                start_pos = match.start()
                start_line = content[:start_pos].count('\n') + 1
                
                # Find matching closing brace
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
                
                elements.append(CodeElement(
                    name=class_name,
                    type='class',
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line
                ))
        
        except Exception as e:
            print(f"Error parsing C++ file {file_path}: {e}")
        
        return elements
