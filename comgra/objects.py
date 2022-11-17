import abc
import collections
import copy
import dataclasses
from typing import List, Dict, Tuple, Optional, Any, Union, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from comgra.recorder import ComgraRecorder


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
    node_name: str
    role_within_node: str
    iteration: int
    configuration_type: str
    type_of_tensor: str
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
class TensorMetadata:
    shape: List[int]
    index_of_batch_dimension: Optional[int]


@dataclasses.dataclass
class Node:
    full_unique_name: str
    type_of_tensor: str
    items_to_record: List[str]


@dataclasses.dataclass
class TensorRecordings:
    training_step_to_iteration_to_configuration_type: Dict[str, Dict[int, str]] = dataclasses.field(default_factory=dict)
    training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records: Dict[int, Dict[str, Dict[Optional[int], Dict[int, Dict[str, Dict[Tuple[str, str, Any], Optional[Union[torch.Tensor, float]]]]]]]] = dataclasses.field(default_factory=dict)
    node_to_role_to_tensor_metadata: Dict[str, Dict[str, TensorMetadata]] = dataclasses.field(default_factory=dict)

    def update_with_more_recordings(self, other: 'TensorRecordings'):
        my_steps = set(self.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records.keys())
        other_steps = set(other.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records.keys())
        assert len(my_steps) + len(other_steps) == len(my_steps | other_steps), \
            f"The training steps overlap.\n{my_steps}\n{other_steps}"
        self.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records.update(
            other.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records
        )
        self.training_step_to_iteration_to_configuration_type.update(
            other.training_step_to_iteration_to_configuration_type
        )
        assert len(self.node_to_role_to_tensor_metadata) == len(other.node_to_role_to_tensor_metadata), \
            f"{len(self.node_to_role_to_tensor_metadata)}\n{len(other.node_to_role_to_tensor_metadata)}"
        for node, role_to_tensor_metadata in self.node_to_role_to_tensor_metadata.items():
            other_role_to_tensor_metadata = other.node_to_role_to_tensor_metadata[node]
            for role, tensor_metadata in role_to_tensor_metadata.items():
                if role in other_role_to_tensor_metadata:
                    other_tensor_metadata = other_role_to_tensor_metadata[role]
                    assert tensor_metadata == other_tensor_metadata
            role_to_tensor_metadata.update(other_role_to_tensor_metadata)


@dataclasses.dataclass
class StatusAndGraph:
    configuration_type: str
    name_to_node: Dict[str, Node]
    types_of_tensor_recordings: List[str]
    nodes: List[str]
    connections: List[List[str]]
    dag_format: List[List[str]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        for name, v in self.name_to_node.items():
            assert name == v.full_unique_name, (name, v.full_unique_name,)

    def build_dag_format(self, recorder: 'ComgraRecorder', name_to_tensor_representation: Dict[str, TensorRepresentation]):
        dependencies_of = collections.defaultdict(list)
        for a in self.connections:
            dependencies_of[a[1]].append(a[0])
        #
        # Sanity check for nodes vs tensors
        #
        node_to_tensor_names = collections.defaultdict(list)
        for tr in name_to_tensor_representation.values():
            assert tr.role_within_node not in node_to_tensor_names[tr.node_name], \
                f"A node has two tensors with the same name in it: {tr.node_name}, {tr.role_within_node}"
            node_to_tensor_names[tr.node_name].append(tr.role_within_node)
        print(node_to_tensor_names)
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
        nodes_list_list.append([k for k, v in name_to_tensor_representation.items() if v.type_of_tensor == 'input'])
        used_nodes = {a: True for a in nodes_list_list[0]}
        nodes_list_for_parameters_and_targets = []
        for prefix in recorder.prefixes_for_grouping_module_parameters_visually:
            next_set_of_nodes = []
            nodes_list_for_parameters_and_targets.append(next_set_of_nodes)
            for a in name_to_tensor_representation.values():
                if a.type_of_tensor == 'parameter' and a.full_unique_name not in used_nodes:
                    if a.full_unique_name.startswith(prefix):
                        used_nodes[a.full_unique_name] = True
                        next_set_of_nodes.append(a.full_unique_name)
        for a in name_to_tensor_representation.values():
            if a.type_of_tensor == 'parameter':
                assert a.full_unique_name in used_nodes, \
                    f"The parameter {a.full_unique_name} is not covered by any of the provided prefixes: " \
                    f"{recorder.prefixes_for_grouping_module_parameters_visually}"
        nodes_list_for_parameters_and_targets.append([k for k, v in name_to_tensor_representation.items() if v.type_of_tensor == 'target'])
        nodes_to_sort = [k for k, v in name_to_tensor_representation.items() if v.type_of_tensor in ['intermediate', 'output']]
        nodes_without_open_dependencies = {k: True for k, v in name_to_tensor_representation.items() if not dependencies_of[k]}
        c = 0
        while c < len(nodes_to_sort):
            debug += 1
            assert debug < 100000, \
                f"Either the graph is too large or I made a programming mistake and this is an endless loop.\n" \
                f"{c}, {len(nodes_to_sort)}\n{[a for a in nodes_list_list if a]}\n{nodes_to_sort}\n" \
                f"{[c for c in nodes_to_sort if c not in [b for a in nodes_list_list for b in a]]}"
            next_set_of_nodes = []
            nodes_list_list.append(next_set_of_nodes)
            for n in nodes_to_sort:
                if n not in nodes_without_open_dependencies:
                    is_now_a_root = True
                    for dependency in dependencies_of[n]:
                        if dependency not in nodes_without_open_dependencies:
                            is_now_a_root = False
                            break
                    if is_now_a_root:
                        next_set_of_nodes.append(n)
                        c += 1
            for n in next_set_of_nodes:
                nodes_without_open_dependencies[n] = True
        assert c == len(nodes_to_sort)
        for list_of_parameters_and_targets in nodes_list_for_parameters_and_targets:
            farthest_possible_index = 0
            for i, nodes in enumerate(nodes_list_list):
                farthest_possible_index = i
                shared_dependencies_of_nodes = {b for a in nodes for b in dependencies_of[a]}
                if any(a in shared_dependencies_of_nodes for a in list_of_parameters_and_targets):
                    break
            nodes_list_list.insert(farthest_possible_index, list_of_parameters_and_targets)
        nodes_list_list.append([k for k, v in name_to_tensor_representation.items() if v.type_of_tensor == 'loss'])
        nodes_list_list = [a for a in nodes_list_list if len(a) > 0]
        for i, nodes0 in enumerate(nodes_list_list):
            for j, nodes1 in enumerate(nodes_list_list):
                if i < j:
                    shared_dependencies_of_nodes = {b for a in nodes0 for b in dependencies_of[a]}
                    assert not any(a in shared_dependencies_of_nodes for a in nodes1), \
                        f"The construction of the DAG is faulty. Probably the easiest way to debug this is " \
                        f"to deactivate this assert and check what the graph looks like. " \
                        f"Some arrows for dependencies should be pointing in the wrong direction."
        #
        # At this point the nodes_list_list contains tensor_names, not node_names.
        # Translate between these, and make sure there are no contradictions.
        #
        used_node_names = {}
        dag_format = []
        for nodes_list in nodes_list_list:
            tmp = []
            dag_format.append(tmp)
            for tensor_name in nodes_list:
                node_name = name_to_tensor_representation[tensor_name].node_name
                if node_name in used_node_names:
                    assert node_name in tmp, \
                        f"The node_name {node_name} is used for two tensors that get sorted in different ways. " \
                        f"This should not be possible if the tensors have the same role / are used " \
                        f"in the same way."
                else:
                    tmp.append(node_name)
                used_node_names[node_name] = True
        self.dag_format = dag_format
        # Also adjust connections to be based on node instead of tensors
        node_level_connections = [
            list(a)
            for a in list(dict.fromkeys([
                (name_to_tensor_representation[a[0]].node_name, name_to_tensor_representation[a[1]].node_name)
                for a in self.connections
            ]))
        ]
        for from_node, to_node in node_level_connections:
            all_from_tensors = [tr.full_unique_name for tr in name_to_tensor_representation.values() if tr.node_name == from_node]
            all_to_tensors = [tr.full_unique_name for tr in name_to_tensor_representation.values() if tr.node_name == to_node]
            for from_tensor in all_from_tensors:
                for to_tensor in all_to_tensors:
                    assert [from_tensor, to_tensor] in self.connections, \
                        f"Not all tensors that were combined into nodes have a connection in common." \
                        f"\n{from_node}, {to_node}\n{from_tensor}, {to_tensor}"
        self.connections = node_level_connections
        assert sum([len(a) for a in dag_format]) == len(self.nodes), \
            (sum([len(a) for a in dag_format]), len(self.nodes), dag_format, self.nodes)


class DecisionMakerForRecordings(abc.ABC):
    pass

    @abc.abstractmethod
    def is_record_on_this_iteration(self, training_step):
        pass


@dataclasses.dataclass
class DecisionMakerForRecordingsHardcoded(DecisionMakerForRecordings):
    fixed_training_steps: set[int]

    def is_record_on_this_iteration(self, training_step):
        return training_step in self.fixed_training_steps


class DecisionMakerForRecordingsExponentialFalloff(DecisionMakerForRecordings):
    maximum_number_of_recordings: int
    current_valid_steps: List
    current_step_size: int = 1

    def __init__(self, maximum_number_of_recordings, starting_step_size):
        super().__init__()
        assert maximum_number_of_recordings > 1
        self.maximum_number_of_recordings = maximum_number_of_recordings
        self.current_valid_steps = []
        self.current_step_size = starting_step_size

    def is_record_on_this_iteration(self, training_step):
        if self.current_step_size * (self.maximum_number_of_recordings - 1) < training_step:
            self.current_step_size *= 2
        self.current_valid_steps = [
            k for k in self.current_valid_steps
            if k % self.current_step_size == 0
        ]
        assert len(self.current_valid_steps) <= self.maximum_number_of_recordings
        return (training_step % self.current_step_size) == 0
