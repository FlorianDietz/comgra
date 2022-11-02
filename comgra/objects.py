import collections
import dataclasses
from typing import List, Dict, Tuple, Optional, Any, Union

import torch


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
    index_of_batch_dimension: Optional[int]
    value_dimensions: List[int]
    recording_type: str
    items_to_record: List[str]
    record_per_batch_index: bool
    is_a_dependency_of: List[str] = dataclasses.field(default_factory=list)

    def get_size_of_tensor(self):
        assert len(self.shape) == 2
        return self.shape[1 - self.index_of_batch_dimension]


@dataclasses.dataclass
class GlobalStatus:
    prefixes_for_grouping_module_parameters: List[str]
    tensor_representations: Dict[str, TensorRepresentation]
    types_of_tensor_recordings: List[str]

    def __post_init__(self):
        for k, v in self.tensor_representations.items():
            assert k == v.full_unique_name, (k, v.full_unique_name)

    def get_all_items_to_record(self):
        res = []
        for tr in self.tensor_representations.values():
            assert len(tr.items_to_record) > 0, tr.full_unique_name
            for item in tr.items_to_record:
                if item in ['single_value', 'mean', 'abs_mean', 'std']:
                    res.append((tr.full_unique_name, item, None))
                elif item == 'neurons':
                    for i in range(tr.get_size_of_tensor()):
                        res.append((tr.full_unique_name, item, i))
                else:
                    raise NotImplementedError(item)
        return res


@dataclasses.dataclass
class DecisionMakerForRecordings:
    fixed_training_steps: set[int]

    def is_record_on_this_iteration(self, training_step):
        return training_step in self.fixed_training_steps


@dataclasses.dataclass
class TensorRecordings:
    training_time_to_type_of_recording_to_batch_index_to_records: Dict[int, Dict[str, Dict[Optional[int], Dict[Tuple[str, str, Any], Optional[Union[torch.Tensor, float]]]]]] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DirectedAcyclicGraph:
    nodes: List[str]
    connections: List[List[str]]
    dag_format: List[List[str]] = dataclasses.field(default_factory=list)

    def build_dag_format(self, global_status: GlobalStatus):
        dependencies_of = collections.defaultdict(list)
        for a in self.connections:
            dependencies_of[a[1]].append(a[0])
        #
        # Logic for grouping:
        # * Inputs first (None of these have dependencies)
        # * One set for each group of parameters in the same module (None of these have dependencies)
        # * Intermediates and Outputs are sorted in DAG logic
        # * Move each set of parameters to the right as far as possible without violating DAG logic
        # * Targets (None of these have dependencies)
        # * Losses
        #
        nodes_list_list = []
        debug = 0
        nodes_list_list.append([k for k, v in global_status.tensor_representations.items() if v.role == 'input'])
        used_nodes = {a: True for a in nodes_list_list[0]}
        nodes_list_for_parameters = []
        for prefix in global_status.prefixes_for_grouping_module_parameters:
            next_set_of_nodes = []
            nodes_list_for_parameters.append(next_set_of_nodes)
            for a in global_status.tensor_representations.values():
                if a.role == 'parameter' and a.full_unique_name not in used_nodes:
                    if a.full_unique_name.startswith(prefix):
                        used_nodes[a.full_unique_name] = True
                        next_set_of_nodes.append(a.full_unique_name)
        for a in global_status.tensor_representations.values():
            if a.role == 'parameter':
                assert a.full_unique_name in used_nodes, \
                    f"The parameter {a.full_unique_name} is not covered by any of the provided prefixes: " \
                    f"{global_status.prefixes_for_grouping_module_parameters}"
        nodes_to_sort = [k for k, v in global_status.tensor_representations.items() if v.role in ['intermediate', 'output']]
        c = 0
        while c < len(nodes_to_sort):
            debug += 1
            assert debug < 1000, \
                "Either the graph is too large or I made a programming mistake and this is an endless loop."
            next_set_of_nodes = []
            nodes_list_list.append(next_set_of_nodes)
            for n in nodes_to_sort:
                if n not in used_nodes:
                    is_now_a_root = True
                    for dependency in dependencies_of[n]:
                        if dependency not in used_nodes:
                            is_now_a_root = False
                            break
                    if is_now_a_root:
                        next_set_of_nodes.append(n)
                        c += 1
            for n in next_set_of_nodes:
                used_nodes[n] = True
        assert c == len(nodes_to_sort)
        for list_of_parameters in nodes_list_for_parameters:
            farthest_possible_index = 0
            for i, nodes in enumerate(nodes_list_list):
                farthest_possible_index = i
                shared_dependencies_of_nodes = {b for a in nodes for b in dependencies_of[a]}
                if any(a in shared_dependencies_of_nodes for a in list_of_parameters):
                    break
            nodes_list_list.insert(farthest_possible_index, list_of_parameters)
        nodes_list_list.append([k for k, v in global_status.tensor_representations.items() if v.role == 'target'])
        nodes_list_list.append([k for k, v in global_status.tensor_representations.items() if v.role == 'loss'])
        nodes_list_list = [a for a in nodes_list_list if len(a) > 0]
        assert sum([len(a) for a in nodes_list_list]) == len(self.nodes)
        for i, nodes0 in enumerate(nodes_list_list):
            for j, nodes1 in enumerate(nodes_list_list):
                if i < j:
                    shared_dependencies_of_nodes = {b for a in nodes0 for b in dependencies_of[a]}
                    assert not any(a in shared_dependencies_of_nodes for a in nodes1), \
                        f"The construction of the DAG is faulty. Probably the easiest way to debug this is " \
                        f"to deactivate this assert and check what the graph looks like. " \
                        f"Some arrows for dependencies should be pointing in the wrong direction."
        self.dag_format = nodes_list_list
