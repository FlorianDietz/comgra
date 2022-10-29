import dataclasses
from typing import List, Dict


@dataclasses.dataclass
class ParameterRepresentation:
    name: str
    full_unique_name: str
    shape: List[int]


@dataclasses.dataclass
class ModuleRepresentation:
    name: str
    full_unique_name: str
    submodules: Dict[str, 'ModuleRepresentation']
    parameters: Dict[str, ParameterRepresentation]


@dataclasses.dataclass
class TensorRepresentation:
    full_unique_name: str
    role: str
    shape: List[int]
    is_a_source_for: List[str] = dataclasses.field(default_factory=list)
