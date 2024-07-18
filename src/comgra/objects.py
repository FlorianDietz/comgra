import abc
import collections
import dataclasses
from typing import List, Dict, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from comgra.recorder import ComgraRecorder

from comgra import utilities

SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS = '*old'


@dataclasses.dataclass(frozen=True)
class ParameterRepresentation:
    name: str
    full_unique_name: str
    shape: List[int]


@dataclasses.dataclass(frozen=True)
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


@dataclasses.dataclass(frozen=True)
class Node:
    full_unique_name: str
    type_of_tensor: str


@dataclasses.dataclass
class TrainingStepConfiguration:
    type_of_execution: str
    modules_and_parameters: Dict[str, ModuleRepresentation]
    graph_configuration_per_iteration: List['GraphConfigurationOfOneIteration'] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class TensorRecordings:
    recordings: Optional[utilities.PseudoDb] = None

    @utilities.runtime_analysis_decorator
    def update_with_more_recordings(self, other: 'TensorRecordings'):
        self.recordings.merge(other.recordings)


@dataclasses.dataclass
class GraphConfigurationOfOneIteration:
    hash_of_node_graph_structure: str
    tensor_graph_structure: 'TensorGraphStructure'


@dataclasses.dataclass
class TensorGraphStructure:
    tensor_references: List[TensorReference]
    tensor_connections: List[Tuple[TensorReference, TensorReference]]


@dataclasses.dataclass
class NodeGraphStructure:
    name_to_node: Dict[str, Node]
    node_connections: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    dag_format: List[List[str]] = dataclasses.field(default_factory=list)
    node_graph_hash: str = "TBD"

    def build_dag_format(
            self, recorder: 'ComgraRecorder',
            tensor_references_to_use_for_this_iteration: Set[TensorReference],
            tensor_reference_to_list_of_dependents: Dict[TensorReference, List[TensorReference]],
            tensor_reference_to_representation: Dict[TensorReference, TensorRepresentation],
            tensor_connections: List[Tuple[TensorReference, TensorReference]],
    ):
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
        # Consistency check for combining tensors into nodes.
        # Make sure that there are no connections between two tensors that belong to the same node.
        # For example, three tensors a->b->c, where a and c are both assigned to the same node.
        for refs in node_to_tensor_references.values():
            for i, ref0 in enumerate(refs):
                for ref1 in refs[i+1:]:
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
        #
        # Logic for grouping:
        # * One set for each group of parameters in the same module (None of these have dependencies)
        # * Sort all calculated, input, target and loss tensors according to DAG logic
        # * Move each set of parameters to the right as far as possible without violating DAG logic
        #   (They do not have dependencies, but it's visually cleaner if they are further to the right,
        #   where they are first used and if they remain grouped together by their names.)
        #
        nodes_list_list = []
        used_nodes = set()
        nodes_list_for_parameters: List[List[TensorReference]] = []
        for prefix in recorder.prefixes_for_grouping_module_parameters_visually:
            next_set_of_nodes = []
            nodes_list_for_parameters.append(next_set_of_nodes)
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
        #
        # The complicated part:
        # Construct the DAG of the dependency graph for all 'input', 'calculated', 'target' and 'loss' tensors
        #
        for ref in used_nodes:
            assert all(other_ref in used_nodes for other_ref in node_to_tensor_references[ref.node_name]), \
                ("Programming error. If a tensor has already been placed at this point, "
                 "then so should all other tensors of the same node.")
        nodes_to_sort = [
            ref for ref in tensor_references_to_use_for_this_iteration
            if tensor_reference_to_representation[ref.get_canonical_reference()].type_of_tensor in ['input', 'calculated', 'target', 'loss']
               and ref not in used_nodes
        ]
        c = 0
        while c < len(nodes_to_sort):
            # Add the item to the graph in order, if possible
            no_nodes_can_be_placed = True
            for types_of_node_to_place_next in [('input',), ('calculated',), ('target',), ('loss',)]:
                next_set_of_nodes = []
                nodes_list_list.append(next_set_of_nodes)
                # Only place a TensorReference if ALL TensorReferences of the same node no longer have dependents.
                # This ensures that a node is placed in only one location in the graph.
                # Note: If tensors of the same node have dependencies between them, we could get a deadlock here.
                # However, we raise an exception in this case earlier in this function.
                for ref in nodes_to_sort:
                    type_of_tensor = tensor_reference_to_representation[ref.get_canonical_reference()].type_of_tensor
                    if ref not in used_nodes and ref not in next_set_of_nodes and \
                            type_of_tensor in types_of_node_to_place_next:
                        all_refs_of_that_node = node_to_tensor_references[ref.node_name]
                        they_all_have_no_dependencies = True
                        for r in all_refs_of_that_node:
                            has_remaining_dependencies = any(
                                a for a in tensor_reference_to_list_of_dependencies[r]
                                if a not in used_nodes
                            )
                            if has_remaining_dependencies:
                                they_all_have_no_dependencies = False
                                break
                        if they_all_have_no_dependencies:
                            for r in all_refs_of_that_node:
                                next_set_of_nodes.append(r)
                                c += 1
                for ref in next_set_of_nodes:
                    used_nodes.add(ref)
                # skip nodes of lower priority until we have no nodes of higher priority to place left
                if next_set_of_nodes:
                    no_nodes_can_be_placed = False
                    break
            if no_nodes_can_be_placed:
                # Note:
                # Some loops caused by add_tensor_connection() can be discovered during graph construction,
                # but this is not guaranteed.
                # The caching mechanism to prevent repeated function calls can also prevent
                # a loop from being discovered, depending on the order in which tensors end up being processed.
                raise ValueError(
                    "The dependency graph can not be constructed. "
                    "Its dependencies contain a circular reference, likely caused by add_tensor_connection(). "
                    "If you did not use add_tensor_connection(), this error should not be possible. "
                    "In that case, please file a bug report."
                )
            # Old error message, before no_nodes_can_be_placed became a feature, not a bug, that can be
            # triggered by misuse of add_tensor_connection().
            # This assert remains here for future debugging purposes only because it prints a lot of useful stuff.
            assert not no_nodes_can_be_placed, \
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
        # Parameters
        # For each list of these, place them as far to the right as is possible
        for list_of_parameters in nodes_list_for_parameters:
            farthest_possible_index = 0
            for i, nodes in enumerate(nodes_list_list):
                shared_dependencies_of_nodes = {b for a in nodes for b in tensor_reference_to_list_of_dependencies[a]}
                if any(a in shared_dependencies_of_nodes for a in list_of_parameters):
                    break
                farthest_possible_index += 1
            nodes_list_list.insert(farthest_possible_index, list_of_parameters)
        # Finalize
        nodes_list_list = [a for a in nodes_list_list if len(a) > 0]
        # Sanity check
        assert sum([len(a) for a in nodes_list_list]) == len(set(b for a in nodes_list_list for b in a)), \
            (sum([len(a) for a in nodes_list_list]), len(set(b for a in nodes_list_list for b in a)))
        assert sum([len(a) for a in nodes_list_list]) == len(tensor_references_to_use_for_this_iteration), \
            (sum([len(a) for a in nodes_list_list]), len(tensor_references_to_use_for_this_iteration))
        for i, nodes0 in enumerate(nodes_list_list):
            for j, nodes1 in enumerate(nodes_list_list):
                if i < j:
                    shared_dependencies_of_nodes = {b for a in nodes0 for b in tensor_reference_to_list_of_dependencies[a]}
                    assert not any(a in shared_dependencies_of_nodes for a in nodes1), \
                        (f"Programming error. This error message should never be visible to end users. "
                         f"The construction of the DAG is faulty. "
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
        node_connections = sorted(list(set(
            (dependency.node_name, dependent.node_name)
            for dependency, dependent in tensor_connections
        )))
        self.node_connections = node_connections
        assert len(self.name_to_node) == len({b for a in self.dag_format for b in a})
        # Sanity check
        for i, node_list_0 in enumerate(nodes_list_list):
            for node_list_1 in nodes_list_list[i+1:]:
                for ref0 in node_list_0:
                    for ref1 in node_list_1:
                        assert ref0.node_name != ref1.node_name, \
                            (f"Programming error. This error message should never be visible to end users. "
                             f"'{ref0.tensor_name}' and '{ref1.tensor_name}' both belong to the same "
                             f"node '{ref0.node_name}' "
                             f"and were sorted in different locations of the graph. If this was caused "
                             f"by a connection between any two tensors of this node, that should have been "
                             f"discovered earlier in this function.")
        #
        # node_graph_hash
        #
        values_to_hash = (self.name_to_node, self.node_connections, self.dag_format)
        self.node_graph_hash = f"node_graph_hash_{utilities.recursive_pseudo_hash(values_to_hash)}"


class DecisionMakerForRecordings(abc.ABC):
    pass

    @abc.abstractmethod
    def is_record_on_this_step(self, training_step, type_of_execution):
        pass

    @abc.abstractmethod
    def mark_recording_on_this_step(self, training_step, type_of_execution):
        pass


@dataclasses.dataclass
class DecisionMakerForRecordingsHardcoded(DecisionMakerForRecordings):
    fixed_training_steps: set[int]

    def is_record_on_this_step(self, training_step, type_of_execution):
        return training_step in self.fixed_training_steps

    def mark_recording_on_this_step(self, training_step, type_of_execution):
        pass


@dataclasses.dataclass
class DecisionMakerForRecordingsFrequencyPerType(DecisionMakerForRecordings):
    min_training_steps_difference: int
    exponential_backoff_factor: float = 1.0
    identifier_to_last_recorded_step_and_min_difference: Dict = dataclasses.field(default_factory=dict)

    def is_record_on_this_step(self, training_step, type_of_execution):
        if type_of_execution is None:
            return False
        assert self.exponential_backoff_factor >= 1.0, self.exponential_backoff_factor
        last_recorded_step, min_difference = self.identifier_to_last_recorded_step_and_min_difference.get(
            type_of_execution, (None, self.min_training_steps_difference)
        )
        if last_recorded_step == training_step:
            return True
        if last_recorded_step is None or training_step >= last_recorded_step + min_difference:
            return True
        return False

    def mark_recording_on_this_step(self, training_step, type_of_execution):
        last_recorded_step, min_difference = self.identifier_to_last_recorded_step_and_min_difference.get(
            type_of_execution, (None, self.min_training_steps_difference)
        )
        min_difference = min_difference * self.exponential_backoff_factor
        self.identifier_to_last_recorded_step_and_min_difference[type_of_execution] = (training_step, min_difference)
