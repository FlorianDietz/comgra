import collections
import dataclasses
import functools
import json
import math
import re
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn as torch_nn

from comgra.objects import DecisionMakerForRecordings, StatusAndGraph, ModuleRepresentation, Node, ParameterRepresentation, TensorRecordings, TensorRepresentation
from comgra import utilities


class ComgraRecorder:

    def __init__(
            self, comgra_root_path, group, trial_id,
            prefixes_for_grouping_module_parameters_visually, prefixes_for_grouping_module_parameters_in_nodes,
            parameters_of_trial, decision_maker_for_recordings,
            comgra_is_active=True, max_num_batch_size_to_record=None,
            max_num_mappings_to_save_at_once_during_serialization=20000,
    ):
        comgra_root_path = Path(comgra_root_path)
        assert comgra_root_path.exists()
        self.comgra_is_active = comgra_is_active
        self.trial_id = trial_id
        self.group_path = comgra_root_path / group
        self.trial_path = self.group_path / 'trials' / trial_id
        self.recordings_path = self.trial_path / 'recordings'
        self.recordings_path.mkdir(parents=True, exist_ok=True)
        self.configuration_type = None
        self.configuration_path = None
        self.prefixes_for_grouping_module_parameters_visually = list(prefixes_for_grouping_module_parameters_visually)
        self.prefixes_for_grouping_module_parameters_in_nodes = list(prefixes_for_grouping_module_parameters_in_nodes)
        assert all(isinstance(a, str) for a in prefixes_for_grouping_module_parameters_visually)
        for i, a in enumerate(self.prefixes_for_grouping_module_parameters_visually):
            for j, b in enumerate(self.prefixes_for_grouping_module_parameters_visually):
                if j >= i:
                    break
                assert not a.startswith(b), \
                    f"Earlier prefixes for visual grouping should be more specific than later ones.\n{a}\n{b}"
        for i, a in enumerate(self.prefixes_for_grouping_module_parameters_in_nodes):
            for j, b in enumerate(self.prefixes_for_grouping_module_parameters_in_nodes):
                if j >= i:
                    break
                assert not a.startswith(b), \
                    f"Earlier prefixes for node grouping should be more specific than later ones.\n{a}\n{b}"
        for i, a in enumerate(self.prefixes_for_grouping_module_parameters_visually):
            for j, b in enumerate(self.prefixes_for_grouping_module_parameters_in_nodes):
                if b.startswith(a):
                    break
                assert not b.startswith(a) or a == b, \
                    f"The most specific visual grouping must not be more specific " \
                    f"than the most specific node grouping.\n{a}\n{b}"
        for j, a in enumerate(self.prefixes_for_grouping_module_parameters_in_nodes):
            assert any(b for b in self.prefixes_for_grouping_module_parameters_visually if a.startswith(b)), \
                f"A prefix for node grouping does not have a prefix for visual grouping that is less restrictive." \
                f"\n{a}"
        self.parameters_of_trial = parameters_of_trial
        self.max_num_batch_size_to_record = max_num_batch_size_to_record
        self.max_num_mappings_to_save_at_once_during_serialization = max_num_mappings_to_save_at_once_during_serialization
        #
        # Things that get updated
        #
        self.set_of_top_level_modules = {}
        self.module_to_name = {}
        self.unique_module_names = {}
        self.unique_parameter_names = {}
        self.parameter_to_representation = {}
        self.decision_maker_for_recordings: DecisionMakerForRecordings = decision_maker_for_recordings
        self.current_stage = 'inactive'
        self.types_of_tensor_recordings = ['forward']
        self.current_type_of_tensor_recording = None
        #
        # Things that are recorded once and then compared to
        #
        self.configuration_type_to_status_and_graph: Dict[str, StatusAndGraph] = {}
        #
        # Per training_step
        #
        self.tensor_recordings: Optional[TensorRecordings] = None
        self.mapping_of_tensors_for_extracting_kpis: Dict[Tuple[int, str, Optional[int], Optional[int], str, str, str], Tuple[torch.Tensor, TensorRepresentation]] = {}
        self.is_training_mode = False
        self.training_step = None
        self.type_of_execution_for_diversity_of_recordings = None
        self.iteration = None
        self.record_all_tensors_per_batch_index_by_default = False
        self.computation_step_to_tensor = {}
        self.tensor_to_name_and_iteration = {}
        self.tensor_name_to_node_name = {}
        self.tensor_name_and_iteration_to_representation: Dict[Tuple[str, int], TensorRepresentation] = {}
        self.current_batch_size = None

    @property
    def recording_is_active(self):
        return self.comgra_is_active and self.is_training_mode and self.decision_maker_for_recordings.is_record_on_this_iteration(
            self.training_step, self.type_of_execution_for_diversity_of_recordings,
        )

    def _verify_uniqueness_of_name(self, name, type_of_name):
        if type_of_name == 'module':
            assert name not in self.unique_module_names
            assert name not in self.unique_parameter_names
            self.unique_module_names[name] = True
        elif type_of_name == 'parameter':
            assert name not in self.unique_module_names
            assert name not in self.unique_parameter_names
            self.unique_parameter_names[name] = True
        else:
            raise ValueError(type_of_name)
        return name

    @utilities.runtime_analysis_decorator
    def track_module(self, module_name, module: torch_nn.Module):
        self._track_module_recursive(module_name, module, self.set_of_top_level_modules, [])

    @utilities.runtime_analysis_decorator
    def _track_module_recursive(self, module_name, module: torch_nn.Module, container, preceding_names: List[str]):
        assert module_name not in container
        assert '.' not in module_name
        assert module not in self.module_to_name
        full_unique_name = '.'.join(preceding_names + [module_name])
        self.module_to_name[module] = full_unique_name
        parameters = {}
        for k, v in module.named_parameters(recurse=False):
            parameters[k] = ParameterRepresentation(
                name=k,
                full_unique_name=self._verify_uniqueness_of_name(f"{full_unique_name}.{k}", 'parameter'),
                shape=list(v.shape),
            )
            self.parameter_to_representation[v] = parameters[k]
        submodules = {}
        for k, v in module.named_children():
            self._track_module_recursive(k, v, submodules, preceding_names + [module_name])
        container[module_name] = ModuleRepresentation(
            name=module_name,
            full_unique_name=self._verify_uniqueness_of_name(full_unique_name, 'module'),
            submodules=submodules,
            parameters=parameters,
        )

    @utilities.runtime_analysis_decorator
    def get_name_of_module(self, module):
        return self.module_to_name[module]

    @utilities.runtime_analysis_decorator
    def start_next_recording(
            self, training_step, current_batch_size, is_training_mode,
            type_of_execution_for_diversity_of_recordings,
            record_all_tensors_per_batch_index_by_default=False,
    ):
        self.is_training_mode = is_training_mode
        self.training_step = training_step
        assert type_of_execution_for_diversity_of_recordings is not None, type_of_execution_for_diversity_of_recordings
        assert type_of_execution_for_diversity_of_recordings != 'any_value', type_of_execution_for_diversity_of_recordings
        assert not type_of_execution_for_diversity_of_recordings.startswith('__'), type_of_execution_for_diversity_of_recordings
        assert re.match(r'^[a-zA-Z0-9-_]+$', type_of_execution_for_diversity_of_recordings)
        self.type_of_execution_for_diversity_of_recordings = type_of_execution_for_diversity_of_recordings
        self.iteration = 0
        self.record_all_tensors_per_batch_index_by_default = record_all_tensors_per_batch_index_by_default
        assert self.current_stage == 'inactive', self.current_stage
        self.current_stage = 'started'
        self.types_of_tensor_recordings = []
        self.current_type_of_tensor_recording = 'forward'
        self.computation_step_to_tensor = {}
        self.tensor_to_name_and_iteration = {}
        self.tensor_name_to_node_name = {}
        self.tensor_name_and_iteration_to_representation = {}
        self.current_batch_size = current_batch_size
        self.tensor_recordings = TensorRecordings()
        self.mapping_of_tensors_for_extracting_kpis = {}

    @utilities.runtime_analysis_decorator
    def register_tensor(
            self, tensor_name, tensor: torch.Tensor,
            index_of_batch_dimension=0,
            is_input=False, is_parameter=False, is_output=False, is_target=False, is_loss=False,
            recording_type=None, record_per_batch_index=None,
            node_name=None, role_within_node=None
    ):
        if not self.recording_is_active:
            return
        assert (1 if is_input else 0) + (1 if is_parameter else 0) + (1 if is_output else 0) + \
               (1 if is_target else 0) + (1 if is_loss else 0) <= 1, tensor_name
        assert not tensor_name.startswith('node__')
        node_name = 'node__' + (tensor_name if node_name is None else node_name)
        role_within_node = tensor_name if role_within_node is None else role_within_node
        # Make sure that gradients are generated and retained for later.
        if not tensor.requires_grad:
            tensor.requires_grad = True
        tensor.retain_grad()
        # Make parameters of this function call consistent with each other
        if is_loss:
            recording_type = 'single_value'
            index_of_batch_dimension = None
            value_dimensions = []
        elif is_parameter:
            recording_type = 'kpis'
            index_of_batch_dimension = None
            value_dimensions = [i for i in range(len(tensor.shape))]
        else:
            assert index_of_batch_dimension is not None
            value_dimensions = [i for i in range(len(tensor.shape)) if i != index_of_batch_dimension]
            if recording_type is None:
                recording_type = 'kpis'
        if index_of_batch_dimension is None:
            assert not record_per_batch_index, \
                f"This tensor has no batch dimension and therefore can't have record_per_batch_index=True: {tensor_name}"
        else:
            assert len(tensor.shape) > index_of_batch_dimension and \
                   tensor.shape[index_of_batch_dimension] == self.current_batch_size, tensor_name
            if record_per_batch_index is None:
                record_per_batch_index = self.record_all_tensors_per_batch_index_by_default
        if recording_type == 'single_value':
            assert index_of_batch_dimension is None
        # Create a TensorRepresentation for the tensor and store various references for later.
        if is_input:
            type_of_tensor = 'input'
        elif is_parameter:
            type_of_tensor = 'parameter'
        elif is_output:
            type_of_tensor = 'output'
        elif is_target:
            type_of_tensor = 'target'
        elif is_loss:
            type_of_tensor = 'loss'
        else:
            type_of_tensor = 'intermediate'
        self.computation_step_to_tensor[tensor.grad_fn] = tensor
        assert tensor not in self.tensor_to_name_and_iteration, \
            f"Tensor is already registered under the name and iteration {self.tensor_to_name_and_iteration[tensor]}\n" \
            f"New name and iteration: {(tensor_name, self.iteration)}"
        self.tensor_to_name_and_iteration[tensor] = (tensor_name, self.iteration)
        assert (tensor_name, self.iteration) not in self.tensor_name_and_iteration_to_representation, \
            f"Two tensors were recorded with the same name in the same iteration. " \
            f"Give your tensors unique names: {(tensor_name, self.iteration)}"
        if recording_type == 'kpis':
            items_to_record = ['mean', 'abs_mean', 'std', 'abs_max']
        elif recording_type == 'neurons':
            items_to_record = ['mean', 'abs_mean', 'std', 'abs_max', 'neurons']
        elif recording_type == 'single_value':
            items_to_record = ['single_value']
        else:
            raise NotImplementedError(recording_type)
        tensor_representation = TensorRepresentation(
            full_unique_name=tensor_name,
            node_name=node_name,
            role_within_node=role_within_node,
            iteration=self.iteration,
            configuration_type=self.configuration_type,
            type_of_tensor=type_of_tensor,
            shape=list(tensor.shape),
            index_of_batch_dimension=index_of_batch_dimension,
            value_dimensions=list(value_dimensions),
            recording_type=recording_type,
            items_to_record=list(items_to_record),
            record_per_batch_index=record_per_batch_index,
        )
        self.tensor_name_and_iteration_to_representation[(tensor_name, self.iteration)] = tensor_representation
        self.tensor_name_to_node_name[tensor_name] = node_name
        # Store the current value of the tensor
        self.store_value_of_tensor(tensor, tensor_representation)

    @utilities.runtime_analysis_decorator
    def store_value_of_tensor(self, tensor: torch.Tensor, tensor_representation: TensorRepresentation):
        tensor_name = tensor_representation.full_unique_name
        value_dimensions = tensor_representation.value_dimensions
        batch_size_to_record = self.current_batch_size if self.max_num_batch_size_to_record is None else min(
            self.current_batch_size, self.max_num_batch_size_to_record)
        if tensor_representation.recording_type == 'single_value':
            self.store_value_of_tensor_helper(
                'has_no_batch_dimension', tensor_representation.iteration,
                tensor_representation, 'single_value', tensor.unsqueeze(dim=0).unsqueeze(dim=1),
            )
        else:
            assert len(value_dimensions) > 0, tensor_name
            for item in tensor_representation.items_to_record:
                if tensor_representation.index_of_batch_dimension is None:
                    expansion_dim = 0
                    assert len(tensor.shape) == len(value_dimensions)
                else:
                    expansion_dim = 1
                    assert len(tensor.shape) == 1 + len(value_dimensions)
                # Aggregate over the value dimensions
                if item == 'mean':
                    val = tensor.mean(dim=value_dimensions).unsqueeze(dim=expansion_dim)
                elif item == 'abs_mean':
                    val = tensor.abs().mean(dim=value_dimensions).unsqueeze(dim=expansion_dim)
                elif item == 'std':
                    val = tensor.std(dim=value_dimensions).unsqueeze(dim=expansion_dim)
                    total_number_of_values_per_batch_index = functools.reduce((lambda x, y: x * y), [tensor.shape[dim] for dim in value_dimensions])
                    if total_number_of_values_per_batch_index == 1:
                        val = torch.zeros(val.shape, device=val.device)
                elif item == 'abs_max':
                    val = torch.amax(tensor.abs(), dim=value_dimensions).unsqueeze(dim=expansion_dim)
                elif item == 'neurons':
                    assert len(value_dimensions) == 1, \
                        "This is not implemented for multi-dimensional tensors, as it is unclear how best to portray that."
                    val = tensor.transpose(0, tensor_representation.index_of_batch_dimension)
                    assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size, \
                        (val.shape, tensor_representation.full_unique_name)
                else:
                    raise NotImplementedError(item)
                # Take the mean over the batch, if possible
                batching_types = ((['has_no_batch_dimension'] if tensor_representation.index_of_batch_dimension is None else ['batch_mean']) +
                                  (['individual_batch_indices'] if tensor_representation.record_per_batch_index else []))
                for batching_type in batching_types:
                    if batching_type == 'batch_mean':
                        assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size
                        val1 = val.mean(dim=0).unsqueeze(dim=0)
                    elif batching_type == 'has_no_batch_dimension':
                        assert len(val.shape) == 1, (val.shape, tensor_representation.full_unique_name)
                        val1 = val.unsqueeze(dim=0)
                    elif batching_type == 'individual_batch_indices':
                        assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size, \
                            (val.shape, self.current_batch_size, batch_size_to_record)
                        val1 = val[0:batch_size_to_record, :]
                    else:
                        assert False, batching_type
                    self.store_value_of_tensor_helper(
                        batching_type, tensor_representation.iteration,
                        tensor_representation, item, val1,
                    )

    @utilities.runtime_analysis_decorator
    def store_value_of_tensor_helper(self, batching_type, iteration, tensor_representation: TensorRepresentation, item, tensor):
        key = (self.training_step, self.current_type_of_tensor_recording, batching_type, iteration, tensor_representation.node_name, tensor_representation.role_within_node, item)
        assert key not in self.mapping_of_tensors_for_extracting_kpis, key
        assert len(tensor.shape) == 2, (tensor.shape, key)
        self.mapping_of_tensors_for_extracting_kpis[key] = (tensor, tensor_representation)

    @utilities.runtime_analysis_decorator
    def start_forward_pass(self, iteration, configuration_type):
        assert self.current_stage in ['started', 'after_iteration'], self.current_stage
        self.current_stage = 'forward'
        self.iteration = iteration
        assert isinstance(configuration_type, str) and re.match(r'^[a-zA-Z0-9-_]+$', configuration_type), configuration_type
        self.configuration_type = configuration_type
        self.configuration_path = self.group_path / 'configs' / configuration_type
        self.tensor_recordings.training_step_to_iteration_to_configuration_type.setdefault(self.training_step, {})[self.iteration] = configuration_type
        self.tensor_recordings.training_step_to_type_of_execution_for_diversity_of_recordings[self.training_step] = self.type_of_execution_for_diversity_of_recordings
        if not self.recording_is_active:
            return

    @utilities.runtime_analysis_decorator
    def start_backward_pass(self):
        assert self.current_stage == 'forward', self.current_stage
        self.current_stage = 'backward'
        if not self.recording_is_active:
            return

    @utilities.runtime_analysis_decorator
    def record_current_gradients(self, name_of_loss_group):
        assert name_of_loss_group not in self.types_of_tensor_recordings
        self.types_of_tensor_recordings.append(name_of_loss_group)
        self.current_type_of_tensor_recording = name_of_loss_group
        if not self.recording_is_active:
            return
        for tensor, k in self.tensor_to_name_and_iteration.items():
            tr = self.tensor_name_and_iteration_to_representation[k]
            gradient = torch.zeros(tensor.shape, device=tensor.device) if tensor.grad is None else tensor.grad
            self.store_value_of_tensor(gradient, tr)

    @utilities.runtime_analysis_decorator
    def finish_iteration(self, sanity_check__verify_graph_and_global_status_equal_existing_file=False):
        assert self.current_stage in ['forward', 'backward'], self.current_stage
        self.current_stage = 'after_iteration'
        self.current_type_of_tensor_recording = 'forward'  # This will be used when the parameters get recorded in traverse_graph_backwards
        if not self.recording_is_active:
            return
        #
        # Go backwards through the computation graph, starting from outputs, targets, and losses.
        # Go back until you encounter an input, or you can't go back anymore.
        #
        step_was_already_encountered_with_parameters = collections.defaultdict(list)

        @utilities.runtime_analysis_decorator
        def traverse_graph_backwards(step, last_encountered_named_tensor_and_iteration):
            if step is None:
                return
            t = None
            if step in self.computation_step_to_tensor:
                assert not hasattr(step, 'variable'), \
                    "This shouldn't be possible. hasattr(step, 'variable') is True if it's a leaf, " \
                    "while computation_step_to_tensor is used for intermediate values."
                t = self.computation_step_to_tensor[step]
            if hasattr(step, 'variable'):
                t = step.variable
                # Register parameters in the graph the first time you encounter them.
                if t in self.parameter_to_representation and t not in self.tensor_to_name_and_iteration:
                    tensor_name = self.parameter_to_representation[t].full_unique_name
                    node_name = next((a for a in self.prefixes_for_grouping_module_parameters_in_nodes if tensor_name.startswith(a)), None)
                    self.register_tensor(tensor_name, t, is_parameter=True, node_name=node_name)
            if t is not None:
                k = self.tensor_to_name_and_iteration[t]
                name_of_this_tensor, iteration = k
                tensor_representation = self.tensor_name_and_iteration_to_representation[k]
                assert iteration == self.iteration or tensor_representation.type_of_tensor == 'parameter', \
                    (name_of_this_tensor, iteration, self.iteration)
                if last_encountered_named_tensor_and_iteration is not None and \
                        last_encountered_named_tensor_and_iteration[0] not in tensor_representation.is_a_dependency_of:
                    dependency_is_from_same_iteration = (tensor_representation.iteration == last_encountered_named_tensor_and_iteration[1])
                    if dependency_is_from_same_iteration:
                        tensor_representation.is_a_dependency_of.append(last_encountered_named_tensor_and_iteration[0])
                assert tensor_representation.iteration is not None
                last_encountered_named_tensor_and_iteration = (name_of_this_tensor, tensor_representation.iteration)
                if tensor_representation.type_of_tensor == 'input':
                    return  # Do not track the graph beyond the inputs, which might go into the previous iteration.
            # Do not traverse the same node more often than necessary. It should be done once per unique parameter.
            # (Otherwise this function can get very expensive.)
            if last_encountered_named_tensor_and_iteration in step_was_already_encountered_with_parameters[step]:
                return
            step_was_already_encountered_with_parameters[step].append(last_encountered_named_tensor_and_iteration)
            # Recurse
            for predecessor, other in step.next_functions:
                traverse_graph_backwards(predecessor, last_encountered_named_tensor_and_iteration)
        final_tensors = [
            t for t, ni in self.tensor_to_name_and_iteration.items()
            if self.tensor_name_and_iteration_to_representation[ni].type_of_tensor in ['output', 'target', 'loss']
               and self.tensor_name_and_iteration_to_representation[ni].iteration == self.iteration
        ]
        for tensor in final_tensors:
            traverse_graph_backwards(tensor.grad_fn, None)
        status_and_graph, name_to_tensor_representation_relevant_for_graph_construction = \
            self.build_global_status_and_graph_and_record_tensor_metadata()
        if self.configuration_type not in self.configuration_type_to_status_and_graph:
            assert sum([len(a.is_a_dependency_of) for a in self.tensor_name_and_iteration_to_representation.values()]) > 0, \
                "No computational graph could be constructed. " \
                "The most common error that could cause this is that gradient computations are turned off."
            #
            # Construct global status and graph information
            #
            status_and_graph.build_dag_format(self, name_to_tensor_representation_relevant_for_graph_construction)
            # Make sure the graph and global_status are only DERIVED once per run
            self.configuration_type_to_status_and_graph[self.configuration_type] = status_and_graph
            # Make sure the graph and global_status are only SAVED once per set of multiple trials.
            self.configuration_path.mkdir(parents=True, exist_ok=True)
            path = self.configuration_path / 'status_and_graph.pkl'
            if not path.exists():
                with open(path, 'wb') as f:
                    pickle.dump(status_and_graph, f)
            #
            # Save trial information
            #
            with open(self.trial_path / 'parameters.json', 'w') as f:
                json.dump(self.parameters_of_trial, f)
        if sanity_check__verify_graph_and_global_status_equal_existing_file:
            #
            # Verify that the result is identical to previous results.
            #
            new_version, name_to_tensor_representation_relevant_for_graph_construction = self.build_global_status_and_graph_and_record_tensor_metadata()
            new_version.build_dag_format(self, name_to_tensor_representation_relevant_for_graph_construction)
            path = self.configuration_path / 'status_and_graph.pkl'
            with open(path, 'rb') as f:
                existing_version: StatusAndGraph = pickle.load(f)
            assert new_version.configuration_type == existing_version.configuration_type
            assert len(new_version.name_to_node) == len(existing_version.name_to_node), \
                f"{self.configuration_type}\n" \
                f"{len(new_version.name_to_node)}, {len(existing_version.name_to_node)}\n" \
                f"{[a for a in new_version.name_to_node if a not in existing_version.name_to_node]}\n" \
                f"{[a for a in existing_version.name_to_node if a not in new_version.name_to_node]}"
            for k1, v1 in new_version.name_to_node.items():
                v2 = dataclasses.asdict(existing_version.name_to_node[k1])
                for kk1, vv1 in dataclasses.asdict(v1).items():
                    vv2 = v2[kk1]
                    vv1 = tuple(vv1) if isinstance(vv1, list) else vv1
                    vv2 = tuple(vv2) if isinstance(vv2, list) else vv2
                    assert vv1 == vv2, f"{k1}\n{kk1}\n{vv1}\n{vv2}"
            assert tuple(new_version.types_of_tensor_recordings) == tuple(existing_version.types_of_tensor_recordings)
            assert len(new_version.dag_format) == len(existing_version.dag_format)
            for a, b in zip(new_version.dag_format, existing_version.dag_format):
                assert len(a) == len(b)
                for c, d in zip(a, b):
                    assert c == d

    @utilities.runtime_analysis_decorator
    def build_global_status_and_graph_and_record_tensor_metadata(self) -> Tuple[StatusAndGraph, Dict]:
        nodes = list(dict.fromkeys([
            v.node_name
            for (name, iteration), v in self.tensor_name_and_iteration_to_representation.items()
            if iteration == self.iteration or v.type_of_tensor == 'parameter'
        ]))
        connections = [
            [dependency, dependent]
            for (dependency, iteration), v in self.tensor_name_and_iteration_to_representation.items()
            for dependent in v.is_a_dependency_of
            if iteration == self.iteration or v.type_of_tensor == 'parameter'
        ]
        name_to_node = {}
        name_to_tensor_representation_relevant_for_graph_construction = {}
        for (tensor_name, iteration), v in self.tensor_name_and_iteration_to_representation.items():
            if iteration == self.iteration or v.type_of_tensor == 'parameter':
                node_name = self.tensor_name_to_node_name[tensor_name]
                if node_name in name_to_node:
                    node = name_to_node[node_name]
                else:
                    node = Node(
                        full_unique_name=node_name,
                        type_of_tensor=v.type_of_tensor,
                        items_to_record=list(v.items_to_record),
                    )
                    name_to_node[node_name] = node
                assert v.type_of_tensor == node.type_of_tensor, \
                    f"Node {node_name} stores tensors with different type_of_tensor." \
                    f"\n{v.type_of_tensor}\n{node.type_of_tensor}"
                name_to_tensor_representation_relevant_for_graph_construction[tensor_name] = v
        status_and_graph = StatusAndGraph(
            configuration_type=self.configuration_type,
            name_to_node=name_to_node,
            types_of_tensor_recordings=list(self.types_of_tensor_recordings),
            nodes=nodes,
            connections=connections,
        )
        return status_and_graph, name_to_tensor_representation_relevant_for_graph_construction

    @utilities.runtime_analysis_decorator
    def finish_batch(self):
        assert self.current_stage == 'after_iteration', self.current_stage
        self.current_stage = 'inactive'
        if not self.recording_is_active:
            return
        #
        # Get the KPIs and serialize them.
        #
        # Notes on how this code works, and why:
        # Convert the TensorRecordings from tensor to float.
        # While doing so, minimize GPU-to-CPU transfers by batching the tensors,
        # but don't batch too many at once to avoid overloading memory and causing a crash.
        # It also appears that a crash is caused more often by pickle than by json,
        # which is why this uses json, even though it's slower.
        # To further reduce the chances of a crash, the data is split into several smaller files.
        # This is done by the same mechanism that also regulates how large the GPU/CPU transfer tensor may get.
        # (max_num_mappings_to_save_at_once_during_serialization)
        # The large number of tensors are chunked and loaded as large tensors to save GPU-CPU bandwidth,
        # and each batch is then saved separately in its own file to avoid a SIGKILL.
        #
        file_number = 0

        def save_recordings_so_far():
            nonlocal file_number
            recordings_path_folder = self.recordings_path / f'{self.training_step}__{self.type_of_execution_for_diversity_of_recordings}'
            recordings_path_folder.mkdir(exist_ok=True)
            with open(recordings_path_folder / f'{file_number}.json', 'w') as f:
                dump_dict = dataclasses.asdict(self.tensor_recordings)
                dump_dict['recordings'] = dump_dict['recordings'].serialize()
                json.dump(dump_dict, f)
            file_number += 1
            self.tensor_recordings.recordings = utilities.PseudoDb(attributes=attributes_for_tensor_recordings)
        batch_size_to_record = self.current_batch_size if self.max_num_batch_size_to_record is None else min(
            self.current_batch_size, self.max_num_batch_size_to_record)
        all_batch_indices = list(range(batch_size_to_record))
        attributes_for_tensor_recordings = [
            'training_step', 'type_of_tensor_recording', 'batch_aggregation', 'iteration',
            'node_name', 'role_within_node', 'record_type', 'item', 'metadata',
        ]
        self.tensor_recordings.recordings = utilities.PseudoDb(attributes=attributes_for_tensor_recordings)
        # Save a preliminary file with information about each tensor.
        # (This is saved in the database just in case the values differ between different iterations, etc.)
        # (For example, the shape of a tensor may in rare cases depend on the iteration.)
        for (tensor_name, iteration), tr in self.tensor_name_and_iteration_to_representation.items():
            for item, metadata, val in [
                ('tensor_shape', None, list(tr.shape)),
                ('index_of_batch_dimension', None, tr.index_of_batch_dimension),
            ]:
                final_key = self.training_step, 'not_applicable', 'not_applicable', iteration, tr.node_name, tr.role_within_node, 'meta_information', item, metadata
                self.tensor_recordings.recordings.add_record(final_key, val)
        save_recordings_so_far()
        # Get the KPIs from tensors
        total_num_mappings = len(self.mapping_of_tensors_for_extracting_kpis)
        all_tensors_to_combine = []
        all_keys_to_process = []
        sanity_check_c = 0
        for i, (key, (tensor, tensor_representation)) in enumerate(self.mapping_of_tensors_for_extracting_kpis.items()):
            # Store the tensors, and remember in what format to retrieve them again later.
            training_step, type_of_tensor_recording, batching_type, iteration, node_name, role_within_node, item = key
            assert training_step == self.training_step
            assert len(tensor.shape) == 2, (tensor.shape, key)
            if batching_type == 'individual_batch_indices':
                batch_values = all_batch_indices
                assert tensor.shape[0] == batch_size_to_record, (tensor.shape, self.current_batch_size, batch_size_to_record)
            else:
                batch_values = [batching_type]
                assert tensor.shape[0] == 1, (tensor.shape, self.current_batch_size, batch_size_to_record, key)
            if item == 'neurons':
                neuron_values = range(tensor.shape[1])
            else:
                neuron_values = [None]
                assert tensor.shape[1] == 1
            all_tensors_to_combine.append(tensor.reshape(-1))
            assert tensor.numel() == len(batch_values) * len(neuron_values), (tensor.shape, len(batch_values), len(neuron_values))
            assert tensor.numel() == tensor.reshape(-1).numel()
            for batch_value in batch_values:
                for neuron_value in neuron_values:
                    metadata = neuron_value
                    final_key = training_step, type_of_tensor_recording, batch_value, iteration, node_name, role_within_node, 'data', item, metadata
                    all_keys_to_process.append(final_key)
            # Combine and retrieve once enough tensors have been accumulated
            if (i + 1) % self.max_num_mappings_to_save_at_once_during_serialization == 0 or i == total_num_mappings - 1:
                combined_tensor = torch.cat(all_tensors_to_combine)
                assert len(combined_tensor.shape) == 1
                list_of_floats = combined_tensor.cpu().tolist()
                assert len(list_of_floats) == len(all_keys_to_process), (len(list_of_floats), len(all_keys_to_process))
                for key_to_process, float_value in zip(all_keys_to_process, list_of_floats):
                    assert isinstance(float_value, float), (float_value, key_to_process)
                    self.tensor_recordings.recordings.add_record(key_to_process, float_value)
                    sanity_check_c += 1
                all_tensors_to_combine = []
                all_keys_to_process = []
                save_recordings_so_far()
        assert len(all_tensors_to_combine) == 0
        total_number_of_tensor_values = sum(t.numel() for t, tr in self.mapping_of_tensors_for_extracting_kpis.values())
        assert sanity_check_c == total_number_of_tensor_values, (sanity_check_c, total_number_of_tensor_values)
