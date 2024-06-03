import abc
import collections
import dataclasses
from typing import List, Dict, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from comgra.recorder import ComgraRecorder

from comgra import utilities

SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS = '*old'

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


@dataclasses.dataclass(frozen=True)
class TensorReference:
    tensor_name: str
    iteration: int
    node_name: str
    role_within_node: str
    is_canonical_reference: bool
    previous_reference: Optional['TensorReference']

    def get_canonical_reference(self) -> 'TensorReference':
        if self.is_canonical_reference:
            return self
        return self.previous_reference.get_canonical_reference()


@dataclasses.dataclass(frozen=True)
class TensorRepresentation:
    original_reference: TensorReference
    configuration_type: str
    type_of_tensor: str
    shape: List[int]
    index_of_batch_dimension: Optional[int]
    value_dimensions: List[int]
    recording_type: str
    items_to_record: List[str]
    record_per_batch_index: bool

    def get_size_of_tensor(self):
        assert len(self.shape) == 2
        return self.shape[1 - self.index_of_batch_dimension]


@dataclasses.dataclass
class Node:
    full_unique_name: str
    type_of_tensor: str


@dataclasses.dataclass
class TensorRecordings:
    training_step_to_type_of_execution: Dict[int, str] = dataclasses.field(default_factory=dict)
    training_step_to_configuration_type: Dict[int, str] = dataclasses.field(default_factory=dict)
    recordings: Optional[utilities.PseudoDb] = None

    @utilities.runtime_analysis_decorator
    def update_with_more_recordings(self, other: 'TensorRecordings'):
        self.recordings.merge(other.recordings)


@dataclasses.dataclass
class StatusAndGraphPerIteration:
    name_to_node: Dict[str, Node]
    nodes: List[str]
    tensor_connections: List[List[TensorReference]] = dataclasses.field(default_factory=list)
    node_connections: List[List[str]] = dataclasses.field(default_factory=list)
    dag_format: List[List[str]] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        for name, v in self.name_to_node.items():
            assert name == v.full_unique_name, (name, v.full_unique_name,)

    def build_dag_format(
            self, recorder: 'ComgraRecorder',
            tensor_references_to_use_for_this_iteration: Set[TensorReference],
            tensor_reference_to_list_of_dependents: Dict[TensorReference, List[TensorReference]],
            tensor_reference_to_representation: Dict[TensorReference, TensorRepresentation],
    ):
        for dependency_ref, dependents in tensor_reference_to_list_of_dependents.items():
            for dependent_ref in dependents:
                dependency_type = tensor_reference_to_representation[dependency_ref.get_canonical_reference()].type_of_tensor
                dependent_type = tensor_reference_to_representation[dependent_ref.get_canonical_reference()].type_of_tensor
                # NOTE: If this does turn out to be a required feature, I will need to change the DAG construction.
                # The tensors of type target and loss are currently only added at the end,
                # and are not part of the DAG sorting algorithm.
                # This may not be too much work to change?
                # --> Don't fix what isn't broken. Do this only if necessary / requested.
                if dependency_type == 'target':
                    assert dependent_type in ['loss'], \
                        (f"The tensor '{dependency_ref.tensor_name}' is registered as a 'target' but "
                         f"the tensor '{dependent_ref.tensor_name}' depends on it and is not of type 'loss'.")
                if dependency_type == 'loss':
                    assert False, \
                        (f"The tensor '{dependency_ref.tensor_name}' is registered as a 'loss' but "
                         f"the tensor '{dependent_ref.tensor_name}' depends on it. "
                         f"Losses can not be used as a dependency of other tensors.")
        #
        # General comment about this function:
        # The DAG construction works by creating a list of lists (node_list_list) where the outer list
        # represents the graph from left to right and the inner lists from top to bottom.
        # During construction, each 'node' actually refers to a TensorReference,
        # These get grouped by their node_name only after the graph is constructed,
        # and at that point it has to be verified that all TensorReferences that share a node_name are in fact
        # in the same inner list, so that they can be grouped.
        # If they are not, this means that the dependencies are inconsistent, implying circular references.
        #
        tensor_reference_to_list_of_dependencies: Dict[TensorReference, List[TensorReference]] = collections.defaultdict(list)
        for k, l in tensor_reference_to_list_of_dependents.items():
            for a in l:
                tensor_reference_to_list_of_dependencies[a].append(k)
        node_to_tensor_references: Dict[str, List[TensorReference]] = collections.defaultdict(list)
        for ref in tensor_references_to_use_for_this_iteration:
            assert ref.role_within_node not in [a.role_within_node for a in node_to_tensor_references[ref.node_name]], \
                f"A node has two tensors with the same role_within_node in it: {ref}"
            assert ref.iteration == recorder.iteration, ref
            node_to_tensor_references[ref.node_name].append(ref)
        #
        # Logic for grouping:
        # * One set for each group of parameters in the same module (None of these have dependencies)
        # * Sort all calculated and input tensors according to DAG logic
        # * Losses
        # * Move each set of parameters and targets to the right as far as possible without violating DAG logic
        #   (They do not have dependencies, but it's visually cleaner if they are further to the right,
        #   where they are first used.)
        #
        nodes_list_list = []
        used_nodes = set()
        nodes_list_for_parameters_and_targets: List[List[TensorReference]] = []
        for prefix in recorder.prefixes_for_grouping_module_parameters_visually:
            next_set_of_nodes = []
            nodes_list_for_parameters_and_targets.append(next_set_of_nodes)
            for ref in tensor_references_to_use_for_this_iteration:
                a = tensor_reference_to_representation[ref.get_canonical_reference()]
                if a.type_of_tensor == 'parameter' and ref not in used_nodes:
                    if ref.tensor_name.startswith(prefix):
                        used_nodes.add(ref)
                        next_set_of_nodes.append(ref)
        for ref in tensor_references_to_use_for_this_iteration:
            a = tensor_reference_to_representation[ref.get_canonical_reference()]
            if a.type_of_tensor == 'parameter':
                assert ref in used_nodes, \
                    (f"The parameter {ref.tensor_name} is not covered by any of the provided prefixes: "
                     f"{recorder.prefixes_for_grouping_module_parameters_visually}")
        nodes_list_for_parameters_and_targets.append([
            ref for ref in tensor_references_to_use_for_this_iteration
            if tensor_reference_to_representation[ref.get_canonical_reference()].type_of_tensor == 'target'
        ])
        # The complicated part:
        # Construct the DAG of the dependency graph for all 'calculated' tensors
        nodes_to_sort = [
            ref for ref in tensor_references_to_use_for_this_iteration
            if tensor_reference_to_representation[ref.get_canonical_reference()].type_of_tensor in ['input', 'calculated']
               and ref not in used_nodes
        ]
        c = 0
        while c < len(nodes_to_sort):
            next_set_of_nodes = []
            nodes_list_list.append(next_set_of_nodes)
            for ref in nodes_to_sort:
                if ref not in used_nodes:
                    has_remaining_dependencies = any(
                        a for a in tensor_reference_to_list_of_dependencies[ref]
                        if a not in used_nodes
                    )
                    if not has_remaining_dependencies:
                        next_set_of_nodes.append(ref)
                        c += 1
            for ref in next_set_of_nodes:
                used_nodes.add(ref)
            assert next_set_of_nodes, \
                (f"Programming error. Can't build dependency graph. "
                 f"There are tensors left to be added to the graph, "
                 f"but they all are marked as having open dependencies. "
                 f"If you encounter this error, please contact us.\n"
                 f"{c}, {len(nodes_to_sort)}, {recorder.iteration}\n"
                 f"The current state of the dependency graph:\n"
                 f"{[[b.tensor_name for b in a] for a in nodes_list_list]}\n"
                 f"List of tensors to be sorted:\n"
                 f"{[a.tensor_name for a in nodes_to_sort]}\n"
                 f"These tensors have not been added to the graph. Their dependencies are listed:\n"
                 f"{ {c.tensor_name: [d.tensor_name for d in tensor_reference_to_list_of_dependencies[c]] for c in nodes_to_sort if c not in [b for a in nodes_list_list for b in a]} }\n")
        assert c == len(nodes_to_sort), (c, len(nodes_to_sort))
        assert sum([len(a) for a in nodes_list_list]) == len(set(b for a in nodes_list_list for b in a)), \
            (sum([len(a) for a in nodes_list_list]), len(set(b for a in nodes_list_list for b in a)))
        # Parameters and targets
        # For each list of these, place them as far to the right as is possible
        for list_of_parameters_and_targets in nodes_list_for_parameters_and_targets:
            farthest_possible_index = 0
            for i, nodes in enumerate(nodes_list_list):
                shared_dependencies_of_nodes = {b for a in nodes for b in tensor_reference_to_list_of_dependencies[a]}
                if any(a in shared_dependencies_of_nodes for a in list_of_parameters_and_targets):
                    break
                farthest_possible_index += 1
            nodes_list_list.insert(farthest_possible_index, list_of_parameters_and_targets)
        # Losses
        nodes_list_list.append([
            ref for ref in tensor_references_to_use_for_this_iteration
            if tensor_reference_to_representation[ref.get_canonical_reference()].type_of_tensor == 'loss'
        ])
        nodes_list_list = [a for a in nodes_list_list if len(a) > 0]
        assert sum([len(a) for a in nodes_list_list]) == len(set(b for a in nodes_list_list for b in a)), \
            (sum([len(a) for a in nodes_list_list]), len(set(b for a in nodes_list_list for b in a)))
        assert sum([len(a) for a in nodes_list_list]) == len(tensor_references_to_use_for_this_iteration), \
            (sum([len(a) for a in nodes_list_list]), len(tensor_references_to_use_for_this_iteration))
        for i, nodes0 in enumerate(nodes_list_list):
            for j, nodes1 in enumerate(nodes_list_list):
                if i < j:
                    shared_dependencies_of_nodes = {b for a in nodes0 for b in tensor_reference_to_list_of_dependencies[a]}
                    assert not any(a in shared_dependencies_of_nodes for a in nodes1), \
                        (f"Programming error. The construction of the DAG is faulty. "
                         f"This error message should never be visible to end users. "
                         f"If you are seeing this, please contact us. "
                         f"Probably the easiest way to debug this is "
                         f"to deactivate this assert and check what the graph looks like. "
                         f"Some arrows for dependencies should be pointing in the wrong direction.\n"
                         f"{nodes0}\n{[a for a in nodes1 if a in shared_dependencies_of_nodes]}\n{shared_dependencies_of_nodes}")
        #
        # At this point the nodes_list_list contains TensorReference objects, not node_names.
        # Translate between these, and make sure there are no contradictions.
        #
        used_node_names = {}
        dag_format = []
        for nodes_list in nodes_list_list:
            tmp = []
            dag_format.append(tmp)
            for ref in nodes_list:
                node_name = ref.node_name
                if node_name not in used_node_names:
                    tmp.append(node_name)
                used_node_names[node_name] = True
        dag_format = [
            sorted(a) for a in dag_format
        ]
        self.dag_format = dag_format
        # Save the connections
        tensor_connections = [
            (dependency, dependent)
            for (dependency, dependents) in tensor_reference_to_list_of_dependents.items()
            for dependent in dependents
        ]
        assert len(tensor_connections) == len(set(tensor_connections)), \
            ("Programming error. This should have no duplicates. "
             "If it does, probably the graph traversal has redundant steps and could be optimized.")
        node_connections = [
            list(a)
            for a in list(dict.fromkeys([
                (dependency.node_name, dependent.node_name)
                for dependency, dependent in tensor_connections
            ]))
        ]
        self.tensor_connections = [list(a) for a in tensor_connections]
        self.node_connections = node_connections
        assert sum([len(a) for a in dag_format]) == len(self.nodes), \
            (sum([len(a) for a in dag_format]), len(self.nodes), dag_format, self.nodes)
        inconsistency_found_but_not_identified = False
        for i, node_list_0 in enumerate(nodes_list_list):
            for node_list_1 in nodes_list_list[i+1:]:
                for ref0 in node_list_0:
                    for ref1 in node_list_1:
                        if ref0.node_name == ref1.node_name:
                            # The same node ended up in two different locations of the graph.
                            # Find out if there is a dependency between them and report it to the user.
                            # For example, three tensors a->b->c, where a and c are both assigned to the same node.
                            dependency_chains = {}  # Stores pairs of references that can be used to backtrack
                            new_sources_to_process = {ref0}
                            debug = 0
                            while new_sources_to_process:
                                debug += 1
                                assert debug < 10000, "Programming error. Infinite loop."
                                next_sources = set()
                                for dependency_ref in new_sources_to_process:
                                    for dependent_ref in tensor_reference_to_list_of_dependents[dependency_ref]:
                                        if dependent_ref not in dependency_chains:
                                            dependency_chains[dependent_ref] = dependency_ref
                                            next_sources.add(dependent_ref)
                                new_sources_to_process = next_sources
                            if ref1 in dependency_chains:
                                list_of_dependencies = []
                                ref = ref1
                                while ref in dependency_chains:
                                    list_of_dependencies.insert(0, ref.tensor_name)
                                    ref = dependency_chains[ref]
                                list_of_dependencies.insert(0, ref.tensor_name)
                                raise ValueError(
                                    f"Error while building the dependency graph. "
                                    f"The node '{ref0.node_name}' was assigned to two different locations. "
                                    f"The cause for this has been identified: This list of tensors "
                                    f"depend on each other, and both the first and last tensor in the list are "
                                    f"assigned to this node:\n{list_of_dependencies}"
                                )
                            # If the above error message does not trigger for at least one pair of references,
                            # then we have an inconsistency that stems from a programming error,
                            # and is not the fault of the user.
                            inconsistency_found_but_not_identified = (ref0, ref1)
        assert not inconsistency_found_but_not_identified, \
            (f"Programming error. '{inconsistency_found_but_not_identified[0].tensor_name}' and "
             f"'{inconsistency_found_but_not_identified[1].tensor_name}' both belong to the same "
             f"node '{inconsistency_found_but_not_identified[0].node_name}' and were sorted in different locations "
             f"of the graph, but no connection between any pair of tensors between these locations could be "
             f"identified.")

@dataclasses.dataclass
class StatusAndGraph:
    configuration_type: str
    modules_and_parameters: Dict[str, ModuleRepresentation]
    iteration_to_data: Dict[int, StatusAndGraphPerIteration] = dataclasses.field(default_factory=dict)


class DecisionMakerForRecordings(abc.ABC):
    pass

    @abc.abstractmethod
    def is_record_on_this_iteration(self, training_step, type_of_execution):
        pass


@dataclasses.dataclass
class DecisionMakerForRecordingsHardcoded(DecisionMakerForRecordings):
    fixed_training_steps: set[int]

    def is_record_on_this_iteration(self, training_step, type_of_execution):
        return training_step in self.fixed_training_steps


@dataclasses.dataclass
class DecisionMakerForRecordingsFrequencyPerType(DecisionMakerForRecordings):
    min_training_steps_difference: int
    exponential_backoff_factor: float = 1.0
    identifier_to_last_recorded_step_and_min_difference: Dict = dataclasses.field(default_factory=dict)

    def is_record_on_this_iteration(self, training_step, type_of_execution):
        if type_of_execution is None:
            return False
        assert self.exponential_backoff_factor >= 1.0, self.exponential_backoff_factor
        last_recorded_step, min_difference = self.identifier_to_last_recorded_step_and_min_difference.get(
            type_of_execution, (None, self.min_training_steps_difference)
        )
        if last_recorded_step == training_step:
            return True
        if last_recorded_step is None or training_step >= last_recorded_step + min_difference:
            min_difference = min_difference * self.exponential_backoff_factor
            self.identifier_to_last_recorded_step_and_min_difference[type_of_execution] = (training_step, min_difference)
            return True
        return False


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

    def is_record_on_this_iteration(self, training_step, type_of_execution):
        if self.current_step_size * (self.maximum_number_of_recordings - 1) < training_step:
            self.current_step_size *= 2
        self.current_valid_steps = [
            k for k in self.current_valid_steps
            if k % self.current_step_size == 0
        ]
        assert len(self.current_valid_steps) <= self.maximum_number_of_recordings
        return (training_step % self.current_step_size) == 0
