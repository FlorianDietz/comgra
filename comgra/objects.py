import collections
import dataclasses
from typing import List, Dict, Tuple


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
    is_a_dependency_of: List[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class DirectedAcyclicGraph:
    nodes: List[str]
    connections: List[List[str]]

    def build_dag_format(self):
        nodes_list = []
        used_nodes = {}
        dependencies_of = collections.defaultdict(list)
        for a in self.connections:
            dependencies_of[a[1]].append(a[0])
        debug = 0
        while len(used_nodes) < len(self.nodes):
            debug += 1
            assert debug < 1000, \
                "Either the graph is too large or I made a programming mistake and this is an endless loop."
            next_set_of_nodes = []
            nodes_list.append(next_set_of_nodes)
            for n in self.nodes:
                if n not in used_nodes:
                    is_now_a_root = True
                    for dependency in dependencies_of[n]:
                        if dependency not in used_nodes:
                            is_now_a_root = False
                            print(3, n, dependency)
                            break
                    if is_now_a_root:
                        next_set_of_nodes.append(n)
            for n in next_set_of_nodes:
                used_nodes[n] = True
        print(nodes_list)
        assert sum([len(a) for a in nodes_list]) == len(self.nodes)
        return nodes_list
