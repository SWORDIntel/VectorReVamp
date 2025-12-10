"""
Test Vector Definition and Management

Defines test vectors for comprehensive module coverage.
Each vector represents a specific test scenario with inputs, expected outputs, and coverage targets.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import json
from pathlib import Path


class TestVectorType(Enum):
    """Types of test vectors"""
    UNIT = "unit"
    INTEGRATION = "integration"
    END_TO_END = "end_to_end"
    EDGE_CASE = "edge_case"
    ERROR_HANDLING = "error_handling"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPATIBILITY = "compatibility"


class TestPriority(Enum):
    """Test priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestVector:
    """Represents a single test vector"""
    
    # Identification
    vector_id: str
    name: str
    description: str
    module_name: str
    
    # Classification
    vector_type: TestVectorType
    priority: TestPriority
    
    # Test definition
    inputs: Dict[str, Any] = field(default_factory=dict)
    expected_outputs: Dict[str, Any] = field(default_factory=dict)
    expected_errors: List[str] = field(default_factory=list)
    
    # Coverage targets
    coverage_targets: List[str] = field(default_factory=list)  # Functions/classes to cover
    coverage_minimum: float = 0.0  # Minimum coverage percentage
    
    # Execution
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    setup_function: Optional[str] = None
    teardown_function: Optional[str] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    
    # Framework-specific data
    framework_config: Dict[str, Any] = field(default_factory=dict)
    
    # Results tracking
    last_run: Optional[str] = None
    last_result: Optional[str] = None
    coverage_achieved: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'vector_id': self.vector_id,
            'name': self.name,
            'description': self.description,
            'module_name': self.module_name,
            'vector_type': self.vector_type.value,
            'priority': self.priority.value,
            'inputs': self.inputs,
            'expected_outputs': self.expected_outputs,
            'expected_errors': self.expected_errors,
            'coverage_targets': self.coverage_targets,
            'coverage_minimum': self.coverage_minimum,
            'preconditions': self.preconditions,
            'postconditions': self.postconditions,
            'setup_function': self.setup_function,
            'teardown_function': self.teardown_function,
            'tags': self.tags,
            'dependencies': self.dependencies,
            'timeout': self.timeout,
            'framework_config': self.framework_config,
            'last_run': self.last_run,
            'last_result': self.last_result,
            'coverage_achieved': self.coverage_achieved,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TestVector':
        """Create from dictionary"""
        return cls(
            vector_id=data['vector_id'],
            name=data['name'],
            description=data['description'],
            module_name=data['module_name'],
            vector_type=TestVectorType(data['vector_type']),
            priority=TestPriority(data['priority']),
            inputs=data.get('inputs', {}),
            expected_outputs=data.get('expected_outputs', {}),
            expected_errors=data.get('expected_errors', []),
            coverage_targets=data.get('coverage_targets', []),
            coverage_minimum=data.get('coverage_minimum', 0.0),
            preconditions=data.get('preconditions', []),
            postconditions=data.get('postconditions', []),
            setup_function=data.get('setup_function'),
            teardown_function=data.get('teardown_function'),
            tags=data.get('tags', []),
            dependencies=data.get('dependencies', []),
            timeout=data.get('timeout'),
            framework_config=data.get('framework_config', {}),
            last_run=data.get('last_run'),
            last_result=data.get('last_result'),
            coverage_achieved=data.get('coverage_achieved'),
        )


class TestVectorRegistry:
    """Registry for managing test vectors"""
    
    def __init__(self):
        self.vectors: Dict[str, TestVector] = {}
        self.by_module: Dict[str, List[str]] = {}
        self.by_type: Dict[TestVectorType, List[str]] = {vt: [] for vt in TestVectorType}
        self.by_priority: Dict[TestPriority, List[str]] = {p: [] for p in TestPriority}
    
    def register(self, vector: TestVector):
        """Register a test vector"""
        self.vectors[vector.vector_id] = vector
        
        # Index by module
        if vector.module_name not in self.by_module:
            self.by_module[vector.module_name] = []
        self.by_module[vector.module_name].append(vector.vector_id)
        
        # Index by type
        self.by_type[vector.vector_type].append(vector.vector_id)
        
        # Index by priority
        self.by_priority[vector.priority].append(vector.vector_id)
    
    def get(self, vector_id: str) -> Optional[TestVector]:
        """Get a test vector by ID"""
        return self.vectors.get(vector_id)
    
    def get_by_module(self, module_name: str) -> List[TestVector]:
        """Get all vectors for a module"""
        vector_ids = self.by_module.get(module_name, [])
        return [self.vectors[vid] for vid in vector_ids]
    
    def get_by_type(self, vector_type: TestVectorType) -> List[TestVector]:
        """Get all vectors of a type"""
        vector_ids = self.by_type.get(vector_type, [])
        return [self.vectors[vid] for vid in vector_ids]
    
    def get_by_priority(self, priority: TestPriority) -> List[TestVector]:
        """Get all vectors of a priority"""
        vector_ids = self.by_priority.get(priority, [])
        return [self.vectors[vid] for vid in vector_ids]
    
    def save(self, filepath: Path):
        """Save registry to JSON file"""
        data = {
            'vectors': {vid: vec.to_dict() for vid, vec in self.vectors.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: Path):
        """Load registry from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        for vid, vec_data in data['vectors'].items():
            vector = TestVector.from_dict(vec_data)
            self.register(vector)
