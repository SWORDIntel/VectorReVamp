"""
Code Embedder for Vector Database

Embeds source code into vector database for similarity search.
Framework-agnostic implementation.
"""

import ast
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


@dataclass
class CodeSegment:
    """Represents a code segment (function, class, module)."""
    id: str
    type: str  # 'function', 'class', 'module'
    name: str
    file_path: str
    start_line: int
    end_line: int
    code: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent_class: Optional[str] = None


@dataclass
class TestTemplate:
    """Represents a reusable test template."""
    id: str
    name: str
    code_type: str  # 'unit', 'integration', 'utility'
    test_code: str
    coverage_patterns: List[str]
    fixtures_used: List[str]
    parametrization: bool


class CodeEmbedder:
    """Embeds source code into vector database."""
    
    def __init__(self, config):
        """
        Initialize code embedder
        
        Args:
            config: HarnessConfig instance
        """
        self.config = config
        self.db_path = config.vector_db_path
        self.client = None
        self.collections = {}
        
        if CHROMADB_AVAILABLE and config.use_vector_db:
            self._init_db()
    
    def _init_db(self):
        """Initialize ChromaDB client and collections."""
        if not CHROMADB_AVAILABLE:
            return
        
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create collections
        self.collections = {
            'code_functions': self.client.get_or_create_collection(
                name='code_functions',
                metadata={"description": "Function-level code embeddings"}
            ),
            'code_classes': self.client.get_or_create_collection(
                name='code_classes',
                metadata={"description": "Class-level code embeddings"}
            ),
            'code_modules': self.client.get_or_create_collection(
                name='code_modules',
                metadata={"description": "Module-level code embeddings"}
            ),
            'test_templates': self.client.get_or_create_collection(
                name='test_templates',
                metadata={"description": "Reusable test patterns"}
            ),
        }
    
    def _generate_embedding(self, code: str) -> List[float]:
        """
        Generate embedding for code.
        Uses simple hash-based embedding by default.
        Can be replaced with code-aware embedding model.
        """
        # Simple hash-based embedding (replace with actual model)
        hash_obj = hashlib.sha256(code.encode())
        # Convert to 384-dim vector (ChromaDB default)
        hash_bytes = hash_obj.digest()
        embedding = []
        for i in range(0, min(len(hash_bytes), 48)):  # 48 bytes = 384 bits / 8
            embedding.append(float(hash_bytes[i]) / 255.0)
        # Pad to 384 dimensions
        while len(embedding) < 384:
            embedding.append(0.0)
        return embedding[:384]
    
    def _parse_python_file(self, file_path: Path) -> List[CodeSegment]:
        """Parse Python file and extract code segments."""
        segments = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    segment = self._extract_function(node, file_path, content)
                    segments.append(segment)
                elif isinstance(node, ast.ClassDef):
                    segment = self._extract_class(node, file_path, content)
                    segments.append(segment)
            
            # Add module-level segment
            rel_path = str(file_path.relative_to(self.config.source_root))
            module_segment = CodeSegment(
                id=f"{rel_path}::module",
                type='module',
                name=file_path.stem,
                file_path=rel_path,
                start_line=1,
                end_line=len(content.splitlines()),
                code=content[:1000],  # First 1000 chars
                docstring=ast.get_docstring(tree)
            )
            segments.append(module_segment)
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
        
        return segments
    
    def _extract_function(self, node: ast.FunctionDef, file_path: Path, content: str) -> CodeSegment:
        """Extract function code segment."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        function_code = '\n'.join(lines[start_line-1:end_line])
        
        # Get parent class if exists
        parent_class = None
        try:
            tree = ast.parse(content)
            for parent in ast.walk(tree):
                if isinstance(parent, ast.ClassDef):
                    for item in parent.body:
                        if item == node:
                            parent_class = parent.name
                            break
        except:
            pass
        
        # Create unique ID
        rel_path = str(file_path.relative_to(self.config.source_root))
        unique_id = f"{rel_path}::{node.name}::{start_line}"
        
        return CodeSegment(
            id=unique_id,
            type='function',
            name=node.name,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            code=function_code,
            docstring=ast.get_docstring(node),
            signature=ast.unparse(node.args) if hasattr(ast, 'unparse') else None,
            parent_class=parent_class
        )
    
    def _extract_class(self, node: ast.ClassDef, file_path: Path, content: str) -> CodeSegment:
        """Extract class code segment."""
        lines = content.splitlines()
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        class_code = '\n'.join(lines[start_line-1:end_line])
        
        rel_path = str(file_path.relative_to(self.config.source_root))
        unique_id = f"{rel_path}::{node.name}::{start_line}"
        
        return CodeSegment(
            id=unique_id,
            type='class',
            name=node.name,
            file_path=rel_path,
            start_line=start_line,
            end_line=end_line,
            code=class_code,
            docstring=ast.get_docstring(node)
        )
    
    def embed_codebase(self):
        """Embed entire codebase into vector database."""
        if not CHROMADB_AVAILABLE or not self.config.use_vector_db:
            print("ChromaDB not available or disabled. Skipping embedding.")
            return
        
        print(f"[*] Embedding codebase from {self.config.source_root}...")
        
        all_segments = []
        source_patterns = self.config.framework.source_patterns
        
        # Find all Python files
        python_files = set()
        for pattern in source_patterns:
            for py_file in self.config.source_root.glob(pattern):
                if '__pycache__' in str(py_file) or py_file.name == "__init__.py":
                    continue
                if py_file.suffix == ".py":
                    python_files.add(py_file)
        
        print(f"[*] Found {len(python_files)} Python files")
        
        for py_file in python_files:
            segments = self._parse_python_file(py_file)
            all_segments.extend(segments)
        
        print(f"[*] Extracted {len(all_segments)} code segments")
        
        # Group by type and embed
        by_type = defaultdict(list)
        for segment in all_segments:
            by_type[segment.type].append(segment)
        
        for seg_type, segments in by_type.items():
            collection_name = f'code_{seg_type}s'
            if collection_name not in self.collections:
                continue
            
            collection = self.collections[collection_name]
            
            print(f"[*] Embedding {len(segments)} {seg_type} segments...")
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for segment in segments:
                ids.append(segment.id)
                embeddings.append(self._generate_embedding(segment.code))
                metadatas.append({
                    'name': segment.name,
                    'file_path': segment.file_path,
                    'start_line': segment.start_line,
                    'end_line': segment.end_line,
                    'has_docstring': bool(segment.docstring),
                    'parent_class': segment.parent_class or '',
                })
                documents.append(segment.code[:500])  # First 500 chars
            
            # Batch add to collection
            batch_size = 100
            for i in range(0, len(ids), batch_size):
                batch_ids = ids[i:i+batch_size]
                batch_embeddings = embeddings[i:i+batch_size]
                batch_metadatas = metadatas[i:i+batch_size]
                batch_documents = documents[i:i+batch_size]
                
                collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                    documents=batch_documents
                )
            
            print(f"[+] Embedded {len(segments)} {seg_type} segments")
        
        print(f"[+] Codebase embedding complete! Total segments: {len(all_segments)}")
    
    def embed_existing_tests(self):
        """Embed existing test files as templates."""
        if not CHROMADB_AVAILABLE or not self.config.use_vector_db:
            print("ChromaDB not available or disabled. Skipping test embedding.")
            return
        
        print(f"[*] Embedding test templates from {self.config.test_dir}...")
        
        test_patterns = self.config.framework.test_patterns
        test_files = set()
        
        for pattern in test_patterns:
            for test_file in self.config.test_dir.glob(pattern):
                if test_file.suffix == ".py":
                    test_files.add(test_file)
        
        print(f"[*] Found {len(test_files)} test files")
        
        templates = []
        
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract test functions
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                        test_code = ast.get_source_segment(content, node)
                        if test_code:
                            rel_path = str(test_file.relative_to(self.config.test_dir))
                            unique_id = f"{rel_path}::{node.name}::{node.lineno}"
                            
                            template = TestTemplate(
                                id=unique_id,
                                name=node.name,
                                code_type=self._classify_test_type(test_code),
                                test_code=test_code,
                                coverage_patterns=[],
                                fixtures_used=self._extract_fixtures(test_code),
                                parametrization='parametrize' in test_code
                            )
                            templates.append(template)
            
            except Exception as e:
                print(f"Error parsing {test_file}: {e}")
        
        print(f"[*] Extracted {len(templates)} test templates")
        
        if templates and 'test_templates' in self.collections:
            collection = self.collections['test_templates']
            
            ids = []
            embeddings = []
            metadatas = []
            documents = []
            
            for template in templates:
                ids.append(template.id)
                embeddings.append(self._generate_embedding(template.test_code))
                metadatas.append({
                    'name': template.name,
                    'code_type': template.code_type,
                    'parametrized': template.parametrization,
                    'fixtures': ','.join(template.fixtures_used),
                })
                documents.append(template.test_code[:500])
            
            collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas,
                documents=documents
            )
            
            print(f"[+] Embedded {len(templates)} test templates")
    
    def _classify_test_type(self, test_code: str) -> str:
        """Classify test type based on code patterns."""
        test_code_lower = test_code.lower()
        
        if 'plugin' in test_code_lower and 'parametrize' in test_code_lower:
            return 'plugin'
        elif 'router' in test_code_lower or 'orchestrat' in test_code_lower:
            return 'orchestrator'
        elif 'integration' in test_code_lower:
            return 'integration'
        else:
            return 'unit'
    
    def _extract_fixtures(self, test_code: str) -> List[str]:
        """Extract fixture names from test code."""
        fixtures = []
        import re
        # Simple pattern matching for fixtures
        pattern = r'def test_\w+\(([^)]+)\)'
        matches = re.findall(pattern, test_code)
        for match in matches:
            params = [p.strip().split(':')[0].strip() for p in match.split(',')]
            fixtures.extend(params)
        return list(set(fixtures))
    
    def find_similar_tests(self, target_code: str, n_results: int = 5) -> List[Dict]:
        """Find similar test templates for target code."""
        if not CHROMADB_AVAILABLE or not self.config.use_vector_db:
            return []
        
        if 'test_templates' not in self.collections:
            return []
        
        # Generate embedding for target code
        embedding = self._generate_embedding(target_code)
        
        # Search in test_templates collection
        collection = self.collections['test_templates']
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results
        )
        return results
