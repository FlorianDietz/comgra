import collections
import dataclasses
import functools
import gzip
import json
import numbers
import os
import re
import shutil
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple

import msgpack

import torch
from torch import nn as torch_nn

from comgra.objects import SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS, \
    DecisionMakerForRecordings, StatusAndGraph, ModuleRepresentation, Node, ParameterRepresentation, TensorRecordings, \
    TensorReference, TensorRepresentation
from comgra import utilities


class ComgraRecorder:

    def __init__(
            self, comgra_root_path, group, trial_id,
            prefixes_for_grouping_module_parameters_visually,
            prefixes_for_grouping_module_parameters_in_nodes,
            decision_maker_for_recordings,
            comgra_is_active=True, max_num_batch_size_to_record=None,
            max_num_mappings_to_save_at_once_during_serialization=20000,
            type_of_serialization='msgpack',
            calculate_svd_and_other_expensive_operations_of_parameters=True,
    ):
        comgra_root_path = Path(comgra_root_path)
        assert comgra_root_path.exists()
        self.comgra_is_active = comgra_is_active
        self.trial_id = trial_id
        self.group_path = comgra_root_path / group
        self.trial_path = self.group_path / 'trials' / trial_id
        self.recordings_path = self.trial_path / 'recordings'
        if comgra_is_active:
            self.recordings_path.mkdir(parents=True, exist_ok=True)
        self.configuration_type = None
        self.configuration_path = None
        self.type_of_serialization = type_of_serialization
        self.calculate_svd_and_other_expensive_operations_of_parameters = calculate_svd_and_other_expensive_operations_of_parameters
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
        self.max_num_batch_size_to_record = max_num_batch_size_to_record
        self.max_num_mappings_to_save_at_once_during_serialization = max_num_mappings_to_save_at_once_during_serialization
        #
        # Things that get updated
        #
        self.set_of_top_level_modules = {}
        self.notes = []
        self.module_to_name = {}
        self.unique_module_names = {}
        self.unique_parameter_names = {}
        self.parameter_to_representation = {}
        self.decision_maker_for_recordings: DecisionMakerForRecordings = decision_maker_for_recordings
        self.current_stage = 'inactive'
        self.types_of_tensor_recordings = ['forward']
        self.current_type_of_tensor_recording = None
        #
        # KPI graph recording
        #
        self.kpi_graph_exponential_backoff_factor = 1.05
        self.kpi_graph_history_to_check_for_outliers = 6
        self.kpi_graph_factor_for_detecting_outliers = 0.5
        self.kpi_graph_excerpt = {}
        self.kpi_graph_changed = False
        self.kpi_graph_next_training_step_to_update_file = 10.0
        #
        # Things that are recorded once and then compared to
        #
        self.configuration_type_to_status_and_graph: Dict[str, StatusAndGraph] = {}
        #
        # Per training_step
        #
        self.tensor_recordings: Optional[TensorRecordings] = None
        self.mapping_of_tensors_for_extracting_kpis: Dict[Tuple[int, str, Optional[int], Optional[int], str, str, str], Tuple[torch.Tensor, TensorRepresentation]] = {}
        self.training_step = None
        self.type_of_execution = None
        self.iteration = None
        self.record_all_tensors_per_batch_index_by_default = False
        self.computation_step_to_tensor = {}
        self.tensor_to_list_of_references: Dict[torch.Tensor, List[TensorReference]] = {}  # The first of these tuples is the canonical one
        self.tensor_reference_to_representation: Dict[TensorReference, TensorRepresentation] = {}
        self.current_batch_size = None
        self.override__recording_is_active = None

    def recording_is_active(self):
        if self.override__recording_is_active is None:
            return self.comgra_is_active and self.decision_maker_for_recordings.is_record_on_this_iteration(
                self.training_step, self.type_of_execution,
            )
        return self.comgra_is_active and self.override__recording_is_active

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
    def add_note(self, note):
        if not self.comgra_is_active:
            return
        self.notes.append(str(note))
        with open(self.trial_path / 'notes.json', 'w') as f:
            json.dump(self.notes, f)

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
            self, training_step, current_batch_size,
            type_of_execution,
            record_all_tensors_per_batch_index_by_default=False,
            override__recording_is_active=None,
    ):
        self.training_step = training_step
        assert type_of_execution is not None, type_of_execution
        assert type_of_execution != 'any_value', type_of_execution
        assert not type_of_execution.startswith('__'), type_of_execution
        assert re.match(r'^[a-zA-Z0-9-_]+$', type_of_execution)
        self.type_of_execution = type_of_execution
        self.iteration = None
        self.record_all_tensors_per_batch_index_by_default = record_all_tensors_per_batch_index_by_default
        assert self.current_stage == 'inactive', self.current_stage
        self.current_stage = 'started'
        self.types_of_tensor_recordings = []
        self.current_type_of_tensor_recording = 'forward'
        self.computation_step_to_tensor = {}
        self.tensor_to_list_of_references = {}
        self.tensor_reference_to_representation = {}
        self.current_batch_size = current_batch_size
        self.tensor_recordings = TensorRecordings()
        self.mapping_of_tensors_for_extracting_kpis = {}
        self.override__recording_is_active = override__recording_is_active

    @utilities.runtime_analysis_decorator
    def register_tensor(
            self, tensor_name, tensor: torch.Tensor,
            index_of_batch_dimension=0,
            is_input=False, is_parameter=False, is_output=None, is_target=False, is_loss=False,
            recording_type=None, record_per_batch_index=None,
            node_name=None, role_within_node=None, is_initial_value=False,
    ):
        if not self.recording_is_active():
            return
        assert self.iteration is not None or is_initial_value, \
            "You must call start_iteration() before registering any tensors, unless you use is_initial_value=True."
        assert not is_initial_value or self.iteration is None, \
            ("You can only use is_initial_value=True when registering a tensor "
             "before the first call of start_iteration().")
        assert not is_initial_value or self.current_stage == 'started', \
            ("Use is_initial_value=True to register the initial value of a hidden state after "
             "calling start_recording() but before start_iteration().")
        iteration_to_use_for_registration = -1 if is_initial_value else self.iteration
        assert (1 if is_input else 0) + (1 if is_parameter else 0) + (1 if is_output else 0) + \
               (1 if is_target else 0) + (1 if is_loss else 0) <= 1, tensor_name
        assert not tensor_name.startswith('node__')
        if is_output is not None:
            utilities.warn_once(
                "Comgra API warning: The is_output parameter of register_tensor() is deprecated. "
                "Outputs are now detected automatically, so this argument can be removed."
            )
        node_name = 'node__' + (tensor_name if node_name is None else node_name)
        role_within_node = tensor_name if role_within_node is None else role_within_node
        # Safety checks to avoid overlap with SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS
        assert '*' not in tensor_name, "tensor_name can't contain an asterisk."
        assert '*' not in node_name, "node_name can't contain an asterisk."
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
            if self.calculate_svd_and_other_expensive_operations_of_parameters and len(tensor.shape) == 2:
                recording_type = 'kpis_and_svd'
            else:
                recording_type = 'kpis'
            index_of_batch_dimension = None
            value_dimensions = [i for i in range(len(tensor.shape))]
            iteration_to_use_for_registration = -1
        else:
            assert index_of_batch_dimension is not None
            value_dimensions = [i for i in range(len(tensor.shape)) if i != index_of_batch_dimension]
            if recording_type is None:
                recording_type = 'neurons'
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
        elif is_target:
            type_of_tensor = 'target'
        elif is_loss:
            type_of_tensor = 'loss'
        else:
            type_of_tensor = 'calculated'
        self.computation_step_to_tensor[tensor.grad_fn] = tensor
        if recording_type == 'kpis':
            items_to_record = ['mean', 'abs_mean', 'std', 'abs_max']
        elif recording_type == 'kpis_and_svd':
            items_to_record = ['mean', 'abs_mean', 'std', 'abs_max', 'svd']
        elif recording_type == 'neurons':
            items_to_record = ['mean', 'abs_mean', 'std', 'abs_max', 'neurons']
        elif recording_type == 'single_value':
            items_to_record = ['single_value']
        else:
            raise NotImplementedError(recording_type)
        assert (tensor_name, iteration_to_use_for_registration) not in [(ref.tensor_name, ref.iteration) for k, refs in self.tensor_to_list_of_references.items() for ref in refs], \
            (f"Two tensors were recorded with the same name in the same iteration. "
             f"Give your tensors unique names: {(tensor_name, iteration_to_use_for_registration)}")
        assert (node_name, role_within_node) not in [
            (ref.node_name, ref.role_within_node)
            for refs in self.tensor_to_list_of_references.values()
            for ref in refs
            if ref.iteration == iteration_to_use_for_registration
        ], \
            (f"For iteration {iteration_to_use_for_registration}, the tensor '{tensor_name}' defined for node '{node_name}' "
             f"has the same role as a previously recorded "
             f"tensor within that same node: '{role_within_node}'")
        tensor_is_registered_for_the_first_time = (tensor not in self.tensor_to_list_of_references)
        previous_reference = (None if tensor_is_registered_for_the_first_time else self.tensor_to_list_of_references[tensor][-1])
        tensor_reference = TensorReference(
            tensor_name=tensor_name,
            iteration=iteration_to_use_for_registration,
            node_name=node_name,
            role_within_node=role_within_node,
            is_canonical_reference=tensor_is_registered_for_the_first_time,
            previous_reference=previous_reference,
        )
        if tensor_is_registered_for_the_first_time:
            self.tensor_to_list_of_references[tensor] = [tensor_reference]
            tensor_representation = TensorRepresentation(
                original_reference=tensor_reference,
                configuration_type=self.configuration_type,
                type_of_tensor=type_of_tensor,
                shape=list(tensor.shape),
                index_of_batch_dimension=index_of_batch_dimension,
                value_dimensions=list(value_dimensions),
                recording_type=recording_type,
                items_to_record=list(items_to_record),
                record_per_batch_index=record_per_batch_index,
            )
            self.tensor_reference_to_representation[tensor_reference] = tensor_representation
            # Store the current value of the tensor
            self.store_value_of_tensor(tensor, tensor_representation)
        else:
            # Every subsequent time the same tensor is recorded, just save the reference to the earlier recording
            # (by storing the name in the same list)
            # and make sure it is consistent with the previous recording
            self.tensor_to_list_of_references[tensor].append(tensor_reference)
            canonical_reference = self.tensor_to_list_of_references[tensor][0]
            tr = self.tensor_reference_to_representation[canonical_reference]
            msg = (f"The tensor '{tensor_name}' on iteration {iteration_to_use_for_registration} "
                   f"was previously recorded with "
                   f"the name '{tr.original_reference.tensor_name}' "
                   f"and iteration {tr.original_reference.iteration}, "
                   f"but with different values for ")
            assert tr.type_of_tensor == type_of_tensor, \
                msg + "its type_of_tensor"
            assert tr.index_of_batch_dimension == index_of_batch_dimension, \
                msg + "its index_of_batch_dimension"
            assert tuple(tr.value_dimensions) == tuple(value_dimensions), \
                msg + "its value_dimensions"
            assert tr.recording_type == recording_type, \
                msg + "its recording_type"
            assert tuple(tr.items_to_record) == tuple(items_to_record), \
                msg + "its items_to_record"
            assert tr.record_per_batch_index == record_per_batch_index, \
                msg + "its record_per_batch_index"
            for ref1 in self.tensor_to_list_of_references[tensor]:
                for ref2 in self.tensor_to_list_of_references[tensor]:
                    if ref1 is not ref2 and ref1.iteration == ref2.iteration:
                        assert ref1.node_name != ref2.node_name, \
                            (f"The tensor '{ref1.tensor_name}' has been recorded twice for the same node "
                             f"({ref1.node_name}) on the same iteration ({ref1.iteration}). "
                             f"This is not allowed because it is ambiguous how to organize this in a graph.")


    @utilities.runtime_analysis_decorator
    def store_value_of_tensor(self, tensor: torch.Tensor, tensor_representation: TensorRepresentation):
        tensor_name = tensor_representation.original_reference.tensor_name
        value_dimensions = tensor_representation.value_dimensions
        batch_size_to_record = self.current_batch_size if self.max_num_batch_size_to_record is None else min(
            self.current_batch_size, self.max_num_batch_size_to_record)
        if tensor_representation.recording_type == 'single_value':
            self.store_value_of_tensor_helper(
                'has_no_batch_dimension', tensor_representation.original_reference.iteration,
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
                elif item == 'svd':
                    val = self.get_highest_svd(tensor)
                elif item == 'neurons':
                    val = torch.movedim(tensor, tensor_representation.index_of_batch_dimension, 0)
                    val = val.reshape((val.shape[0], -1))
                    assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size, \
                        (val.shape, tensor_representation.original_reference)
                else:
                    raise NotImplementedError(item)
                # Take aggregates over the batch, if possible
                batching_types = ((['has_no_batch_dimension'] if tensor_representation.index_of_batch_dimension is None else ['batch_mean', 'batch_abs_max', 'batch_std']) +
                                  (['individual_batch_indices'] if tensor_representation.record_per_batch_index else []))
                for batching_type in batching_types:
                    if batching_type == 'batch_mean':
                        assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size
                        val1 = val.mean(dim=0).unsqueeze(dim=0)
                    elif batching_type == 'batch_std':
                        assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size
                        val1 = val.std(dim=0).unsqueeze(dim=0)
                    elif batching_type == 'batch_abs_max':
                        assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size
                        val1 = val.abs().max(dim=0)[0].unsqueeze(dim=0)
                    elif batching_type == 'has_no_batch_dimension':
                        assert len(val.shape) == 1, (val.shape, tensor_representation.original_reference)
                        val1 = val.unsqueeze(dim=0)
                    elif batching_type == 'individual_batch_indices':
                        assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size, \
                            (val.shape, self.current_batch_size, batch_size_to_record)
                        val1 = val[0:batch_size_to_record, :]
                    else:
                        assert False, batching_type
                    self.store_value_of_tensor_helper(
                        batching_type, tensor_representation.original_reference.iteration,
                        tensor_representation, item, val1,
                    )

    @utilities.runtime_analysis_decorator
    def get_highest_svd(self, tensor):
        try:
            res = torch.linalg.svdvals(tensor)[:1]
        except:
            res = torch.tensor([float('nan')], device=tensor.device)
        return res

    @utilities.runtime_analysis_decorator
    def store_value_of_tensor_helper(self, batching_type, iteration, tensor_representation: TensorRepresentation, item, tensor):
        key = (
            self.training_step, self.current_type_of_tensor_recording, batching_type, iteration,
            tensor_representation.original_reference.node_name,
            tensor_representation.original_reference.role_within_node,
            item,
        )
        assert key not in self.mapping_of_tensors_for_extracting_kpis, key
        assert len(tensor.shape) == 2, (tensor.shape, key)
        self.mapping_of_tensors_for_extracting_kpis[key] = (tensor, tensor_representation)

    @utilities.runtime_analysis_decorator
    def start_iteration(self, configuration_type):
        assert self.current_stage in ['started', 'after_iteration'], self.current_stage
        self.current_stage = 'forward'
        self.iteration = 0 if self.iteration is None else (self.iteration + 1)
        assert isinstance(configuration_type, str) and re.match(r'^[a-zA-Z0-9-_,.]+$', configuration_type), configuration_type
        self.configuration_type = configuration_type
        self.configuration_path = self.group_path / 'configs' / configuration_type
        self.tensor_recordings.training_step_to_iteration_to_configuration_type.setdefault(self.training_step, {})[self.iteration] = configuration_type
        self.tensor_recordings.training_step_to_type_of_execution[self.training_step] = self.type_of_execution
        if not self.recording_is_active():
            return

    @utilities.runtime_analysis_decorator
    def start_backward_pass(self):
        assert self.current_stage == 'forward', self.current_stage
        self.current_stage = 'backward'
        if not self.recording_is_active():
            return

    @utilities.runtime_analysis_decorator
    def record_current_gradients(self, name_of_loss_group, set_gradients_to_zero_if_not_a_parameter=False):
        if not self.recording_is_active():
            return
        assert name_of_loss_group not in self.types_of_tensor_recordings, \
            (name_of_loss_group, self.types_of_tensor_recordings)
        self.types_of_tensor_recordings.append(name_of_loss_group)
        self.current_type_of_tensor_recording = name_of_loss_group
        for tensor, refs in self.tensor_to_list_of_references.items():
            tr = self.tensor_reference_to_representation[refs[0]]
            gradient = torch.zeros(tensor.shape, device=tensor.device) if tensor.grad is None else tensor.grad
            self.store_value_of_tensor(gradient, tr)
            if set_gradients_to_zero_if_not_a_parameter and tr.type_of_tensor != 'parameter':
                tensor.grad = None

    @utilities.runtime_analysis_decorator
    def finish_iteration(self, sanity_check__verify_graph_and_global_status_equal_existing_file=True):
        """
        :param sanity_check__verify_graph_and_global_status_equal_existing_file:
            Specify whether you want to run a sanity check to make sure that you specified
            configuration_type of start_iteration() correctly.
            This costs extra time to compute, but if you skip this sanity check,
            you might not realize that you are recording two different computational
            graphs under the same name, and this will lead to errors in the visualization later.
        :return: 
        """
        assert self.current_stage in ['forward', 'backward'], self.current_stage
        self.current_stage = 'after_iteration'
        self.current_type_of_tensor_recording = 'forward'  # This will be used when the parameters get recorded in traverse_graph_backwards
        if not self.recording_is_active():
            return
        #
        # Go backwards through the computation graph, starting from outputs, targets, and losses.
        # Go back until you encounter an input, or you can't go back anymore.
        #
        steps_that_have_been_processed = set()
        tensor_references_to_use_for_this_iteration = set()
        tensor_reference_to_list_of_dependents: Dict[TensorReference, List[TensorReference]] = collections.defaultdict(list)

        @utilities.runtime_analysis_decorator
        def traverse_graph_backwards(step, last_encountered_reference):
            """
            This function sets tensor_reference.is_a_dependency_of for each tensor.
            It also registers any parameters that haven't been registered yet upon encountering them for the first time.
            """
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
                if t in self.parameter_to_representation and t not in self.tensor_to_list_of_references:
                    tensor_name = self.parameter_to_representation[t].full_unique_name
                    node_name = next((a for a in self.prefixes_for_grouping_module_parameters_in_nodes if tensor_name.startswith(a)), None)
                    self.register_tensor(tensor_name, t, is_parameter=True, node_name=node_name)
            keep_recursing = True
            if t is not None:
                # Get the first and the last reference to this tensor on this iteration.
                # Or if there is none on this iteration, create a new reference.
                refs = self.tensor_to_list_of_references[t]
                tmp = [ref for ref in refs if ref.iteration == self.iteration]
                if not tmp:
                    original_ref = refs[0]
                    self.tensor_to_list_of_references[t].append(TensorReference(
                        tensor_name=original_ref.tensor_name + SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS,
                        iteration=self.iteration,
                        node_name=original_ref.node_name + SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS,
                        # node_name=original_ref.node_name + f'__reuse_iteration_{original_ref.iteration}' + SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS,
                        role_within_node=original_ref.role_within_node + f'__from_iteration_{original_ref.iteration}',
                        is_canonical_reference=False,
                        previous_reference=self.tensor_to_list_of_references[t][-1],
                    ))
                    refs = self.tensor_to_list_of_references[t]
                    tmp = [ref for ref in refs if ref.iteration == self.iteration]
                    assert len(tmp) == 1
                for ref in tmp:
                    tensor_references_to_use_for_this_iteration.add(ref)
                first_ref = tmp[0]
                last_ref = tmp[-1]
                tensor_representation = self.tensor_reference_to_representation[refs[0]]
                # Set dependencies from last_encountered_named_tensor_and_iteration to last_ref
                if last_encountered_reference is not None:
                    assert last_encountered_reference not in tensor_reference_to_list_of_dependents[last_ref], \
                        ("Programming error. If a dependency is registered twice, that means the graph traversal "
                         "has some redundancy and could be optimized.")
                    tensor_reference_to_list_of_dependents[last_ref].append(last_encountered_reference)
                # If this step is encountered for the first time, connect the references and recurse
                step_has_been_processed = step in steps_that_have_been_processed
                if step_has_been_processed:
                    keep_recursing = False
                else:
                    steps_that_have_been_processed.add(step)
                    # Set dependencies between all references of this tensor that will be displayed at the same time
                    for ref1, ref2 in zip(tmp[:-1], tmp[1:]):
                        tensor_reference_to_list_of_dependents[ref1].append(ref2)
                        assert len(tensor_reference_to_list_of_dependents[ref1]) == 1, \
                            ("This should only be done once and each additional reference to the same tensor should "
                             "just form part of an uninterrupted chain.")
                    # Set the last_encountered_reference and recurse
                    # unless the iteration is earlier than the current iteration or type_of_tensor == 'input'
                    keep_recursing = ((not first_ref.node_name.endswith(
                        SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS))
                                      and (tensor_representation.type_of_tensor != 'input'))
                    last_encountered_reference = first_ref
            if keep_recursing:
                for predecessor, other in step.next_functions:
                    traverse_graph_backwards(predecessor, last_encountered_reference)
        # Backpropagate from any tensor that was created or otherwise referenced in this iteration
        tensors_to_show = [
            tensor for tensor, refs in self.tensor_to_list_of_references.items()
            for ref in refs
            if ref.iteration == self.iteration
        ]
        for tensor in tensors_to_show:
            traverse_graph_backwards(tensor.grad_fn, None)
        # Build the dependency graph based on the data extracted while recursing through the computation graph
        status_and_graph = self.build_global_status_and_graph(
            tensor_references_to_use_for_this_iteration, tensor_reference_to_list_of_dependents,
        )
        if self.configuration_type not in self.configuration_type_to_status_and_graph:
            assert any([ref1 for ref1, refs in tensor_reference_to_list_of_dependents.items() if
                        any([ref2 for ref2 in refs if ref1.get_canonical_reference() != ref2.get_canonical_reference()])]), \
                "No computational graph could be constructed. " \
                "The most common error that could cause this is that gradient computations are turned off."
            #
            # Construct global status and graph information
            #
            status_and_graph.build_dag_format(
                self, tensor_references_to_use_for_this_iteration, tensor_reference_to_list_of_dependents,
                self.tensor_reference_to_representation,
            )
            # Make sure the graph and global_status are only DERIVED once per run
            self.configuration_type_to_status_and_graph[self.configuration_type] = status_and_graph
            # Make sure the graph and global_status are only SAVED once per set of multiple trials.
            self.configuration_path.mkdir(parents=True, exist_ok=True)
            path = self.configuration_path / 'status_and_graph.pkl'
            if not path.exists():
                with open(path, 'wb') as f:
                    pickle.dump(status_and_graph, f)
        if sanity_check__verify_graph_and_global_status_equal_existing_file:
            #
            # Verify that the result is identical to previous results.
            #
            new_version = self.build_global_status_and_graph(
                tensor_references_to_use_for_this_iteration, tensor_reference_to_list_of_dependents,
            )
            new_version.build_dag_format(
                self, tensor_references_to_use_for_this_iteration, tensor_reference_to_list_of_dependents,
                self.tensor_reference_to_representation,
            )
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
            assert tuple(new_version.types_of_tensor_recordings) == tuple(existing_version.types_of_tensor_recordings), \
                f"{tuple(new_version.types_of_tensor_recordings)}\n" \
                f"{tuple(existing_version.types_of_tensor_recordings)}"
            assert len(new_version.dag_format) == len(existing_version.dag_format)
            for a, b in zip(new_version.dag_format, existing_version.dag_format):
                assert len(a) == len(b)
                for c, d in zip(a, b):
                    assert c == d

    @utilities.runtime_analysis_decorator
    def build_global_status_and_graph(
            self,
            tensor_references_to_use_for_this_iteration,
            tensor_reference_to_list_of_dependents,
    ) -> StatusAndGraph:
        for ref1 in tensor_references_to_use_for_this_iteration:
            for ref2 in tensor_references_to_use_for_this_iteration:
                if ref1.get_canonical_reference() == ref2.get_canonical_reference():
                    assert ((ref1.iteration < self.iteration) and (ref1 == ref2)) or \
                           (ref1.iteration == self.iteration), \
                        ("Programming error. All references to the same tensor should belong to the same iteration. "
                         "Either multiple from this one, or one from a previous one. "
                         "If this assertion fails, then probably traverse_graph_backwards() has a bug")
        nodes = list(set([a.node_name for a in tensor_references_to_use_for_this_iteration]))
        connections = [
            (dependency, dependent)
            for (dependency, dependents) in tensor_reference_to_list_of_dependents.items()
            for dependent in dependents
        ]
        assert len(connections) == len(set(connections)), \
            ("Programming error. This should have no duplicates. "
             "If it does, probably the graph traversal has redundant steps and could be optimized.")
        name_to_node = {}
        for ref in tensor_references_to_use_for_this_iteration:
            tr = self.tensor_reference_to_representation[ref.get_canonical_reference()]
            node_name = ref.node_name
            if node_name in name_to_node:
                node = name_to_node[node_name]
            else:
                node = Node(
                    full_unique_name=node_name,
                    type_of_tensor=tr.type_of_tensor,
                    items_to_record=list(tr.items_to_record),
                )
                name_to_node[node_name] = node
            assert tr.type_of_tensor == node.type_of_tensor, \
                f"Node {node_name} stores tensors with different type_of_tensor." \
                f"\n{tr.type_of_tensor}\n{node.type_of_tensor}"
        status_and_graph = StatusAndGraph(
            configuration_type=self.configuration_type,
            modules_and_parameters=self.set_of_top_level_modules,
            name_to_node=name_to_node,
            types_of_tensor_recordings=list(self.types_of_tensor_recordings),
            nodes=nodes,
            tensor_connections=[list(a) for a in connections],
        )
        return status_and_graph

    @utilities.runtime_analysis_decorator
    def finish_recording(self):
        assert self.current_stage == 'after_iteration' or not self.recording_is_active(), self.current_stage
        self.current_stage = 'inactive'
        if not self.recording_is_active():
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
            recordings_path_folder = self.recordings_path / f'{self.training_step}__{self.type_of_execution}'
            recordings_path_folder.mkdir(exist_ok=True)
            dump_dict = dataclasses.asdict(self.tensor_recordings)
            dump_dict['recordings'] = dump_dict['recordings'].serialize()
            self.save_file(dump_dict, recordings_path_folder, file_number)
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
        assert len(self.tensor_reference_to_representation) == len(set([(tr.original_reference.tensor_name, tr.original_reference.iteration) for tr in self.tensor_reference_to_representation.values()])), \
            (f"Programming error. TensorRepresentations are saved in duplicates, which may clog the database.\n"
             f"{len(self.tensor_reference_to_representation), len(set([(tr.original_reference.tensor_name, tr.original_reference.iteration) for tr in self.tensor_reference_to_representation.values()]))}")
        for main_ref, tr in self.tensor_reference_to_representation.items():
            for item, metadata, val in [
                ('tensor_shape', None, list(tr.shape)),
                ('index_of_batch_dimension', None, tr.index_of_batch_dimension),
            ]:
                final_key = self.training_step, 'not_applicable', 'not_applicable', main_ref.iteration, main_ref.node_name, main_ref.role_within_node, 'meta_information', item, metadata
                self.add_tensor_recordings_for_key_and_register_alternate_references(final_key, val, main_ref)
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
                neuron_values = []
                shape_without_batch_dimension = list(tensor_representation.shape)
                del shape_without_batch_dimension[tensor_representation.index_of_batch_dimension]

                def rec_helper(buffer):
                    if len(buffer) == len(shape_without_batch_dimension):
                        neuron_values.append(', '.join(map(str, buffer)))
                        return
                    buffer.append(None)
                    for j in range(shape_without_batch_dimension[len(buffer) - 1]):
                        buffer[-1] = j
                        rec_helper(buffer)
                    del buffer[:-1]

                rec_helper([])
            else:
                neuron_values = [None]
                assert tensor.shape[1] == 1
            all_tensors_to_combine.append(tensor.reshape(-1))
            assert tensor.numel() == len(batch_values) * len(neuron_values), (tensor.shape, len(batch_values), len(neuron_values))
            assert tensor.numel() == tensor.reshape(-1).numel()
            for batch_value in batch_values:
                for neuron_value in neuron_values:
                    metadata = neuron_value
                    main_ref = tensor_representation.original_reference
                    final_key = training_step, type_of_tensor_recording, batch_value, iteration, node_name, role_within_node, 'data', item, metadata
                    all_keys_to_process.append((final_key, main_ref))
            # Combine and retrieve once enough tensors have been accumulated
            if (i + 1) % self.max_num_mappings_to_save_at_once_during_serialization == 0 or i == total_num_mappings - 1:
                combined_tensor = torch.cat(all_tensors_to_combine)
                assert len(combined_tensor.shape) == 1
                list_of_floats = combined_tensor.cpu().tolist()
                assert len(list_of_floats) == len(all_keys_to_process), (len(list_of_floats), len(all_keys_to_process))
                for (key_to_process, main_ref), float_value in zip(all_keys_to_process, list_of_floats):
                    assert isinstance(float_value, float), (float_value, key_to_process)
                    self.add_tensor_recordings_for_key_and_register_alternate_references(key_to_process, float_value, main_ref)
                    sanity_check_c += 1
                all_tensors_to_combine = []
                all_keys_to_process = []
                save_recordings_so_far()
        assert len(all_tensors_to_combine) == 0
        total_number_of_tensor_values = sum(t.numel() for t, tr in self.mapping_of_tensors_for_extracting_kpis.values())
        assert sanity_check_c == total_number_of_tensor_values, (sanity_check_c, total_number_of_tensor_values)
        #
        # Save the graph of KPIs, which is independent of the rest of the recordings
        #
        self.save_recorded_kpi_graphs_if_needed()

    def add_tensor_recordings_for_key_and_register_alternate_references(self, key, float_value, ref: TensorReference):
        # Sanity check
        assert sum([
            1 if attr in (tmp:={
                'node_name': ref.node_name,
                'role_within_node': ref.role_within_node,
                'iteration': ref.iteration,
            }) and tmp[attr] == val else 0
            for attr, val in zip(self.tensor_recordings.recordings.attributes, key)
        ]) == 3, (f"The reference does not match the key\n{self.tensor_recordings.recordings.attributes}\n"
                  f"{key}\n{ref}")
        # Register the value
        self.tensor_recordings.recordings.add_record_value(key, float_value)
        # Find alternate references
        tmp = [a for a in self.tensor_to_list_of_references.values() if a[0] is ref]
        assert len(tmp) == 1
        alternate_references = tmp[0][1:]
        # Iterate over all alternate references
        for alt_ref in alternate_references:
            alt_key = tuple([
                {
                    'node_name': alt_ref.node_name,
                    'role_within_node': alt_ref.role_within_node,
                    'iteration': alt_ref.iteration,
                }.get(attr, val)
                for attr, val in zip(self.tensor_recordings.recordings.attributes, key)
            ])
            self.tensor_recordings.recordings.add_record_redirection(alt_key, key)

    @utilities.runtime_analysis_decorator
    def save_file(self, dump_dict, recordings_path_folder, file_number):
        if self.type_of_serialization == 'json':
            return self.save_json(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'zip_json':
            return self.save_zip_json(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'pkl':
            return self.save_pickle(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'msgpack':
            return self.save_msgpack(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'zip_msgpack':
            return self.save_zip_msgpack(dump_dict, recordings_path_folder, file_number)
        else:
            raise ValueError(self.type_of_serialization)

    @utilities.runtime_analysis_decorator
    def save_json(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.json'
        with open(path, 'w') as f:
            json.dump(dump_dict, f)
        return path

    @utilities.runtime_analysis_decorator
    def save_zip_json(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.zip_json'
        with gzip.open(path, 'w') as fout:
            json_bytes = (json.dumps(dump_dict) + "\n").encode('utf-8')
            fout.write(json_bytes)
        return path

    @utilities.runtime_analysis_decorator
    def save_pickle(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(dump_dict, f)
        return path

    @utilities.runtime_analysis_decorator
    def save_msgpack(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.msgpack'
        with open(path, 'wb') as f:
            msgpack.dump(dump_dict, f)
        return path

    @utilities.runtime_analysis_decorator
    def save_zip_msgpack(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.zip_msgpack'
        with gzip.open(path, 'wb') as fout:
            fout.write(msgpack.dumps(dump_dict))
        return path

    @utilities.runtime_analysis_decorator
    def record_kpi_in_graph(self, kpi_group, kpi_name, val, timepoint):
        stats = self.kpi_graph_excerpt.setdefault(kpi_group, {}).setdefault(self.type_of_execution, {}).setdefault(kpi_name, {
            'vals': [],
            'next_timepoint': 0,
        })
        if len(stats['vals']) == self.kpi_graph_history_to_check_for_outliers:
            history_to_check = [a['val'] for a in stats['vals'][-self.kpi_graph_history_to_check_for_outliers:]]
            max_ = max(history_to_check)
            min_ = min(history_to_check)
            dist = max_ - min_
            is_outlier = (val > max_ + dist * self.kpi_graph_factor_for_detecting_outliers
                          or val < min_ - dist * self.kpi_graph_factor_for_detecting_outliers)
        else:
            is_outlier = False
        if timepoint >= stats['next_timepoint'] or is_outlier:
            if isinstance(val, torch.Tensor):
                val = val.item()
            assert isinstance(val, numbers.Number)
            stats['vals'].append({
                'timepoint': timepoint,
                'val': val,
            })
            while timepoint >= stats['next_timepoint']:
                stats['next_timepoint'] = max([1, stats['next_timepoint'] * self.kpi_graph_exponential_backoff_factor])
            self.kpi_graph_changed = True

    @utilities.runtime_analysis_decorator
    def save_recorded_kpi_graphs_if_needed(self):
        if not self.kpi_graph_changed:
            return
        if self.training_step < self.kpi_graph_next_training_step_to_update_file:
            return
        self.kpi_graph_changed = False
        self.kpi_graph_next_training_step_to_update_file *= self.kpi_graph_exponential_backoff_factor
        # Save the file with a tmp suffix first, then overwrite the real one.
        # This prevents issues in case the visualizer is accessing the file while it is being overwritten.
        tmp_path = self.save_file(self.kpi_graph_excerpt, self.trial_path, 'kpi_graph_tmp')
        real_path = tmp_path.parent / f'kpi_graph{tmp_path.suffix}'
        os.replace(tmp_path, real_path)
