import collections
import dataclasses
import json
import re
from pathlib import Path
import pickle
from typing import List, Dict, Optional, Tuple

import torch
from torch import nn as torch_nn

from comgra.objects import DecisionMakerForRecordings, StatusAndGraph, ModuleRepresentation, ParameterRepresentation, TensorRecordings, TensorRepresentation


class ComgraRecorder:

    def __init__(
            self, comgra_root_path, group, trial_id, prefixes_for_grouping_module_parameters, parameters_of_trial,
            decision_maker_for_recordings, comgra_is_active=True
    ):
        comgra_root_path = Path(comgra_root_path)
        assert comgra_root_path.exists()
        self.comgra_is_active = comgra_is_active
        self.trial_id = trial_id
        self.group_path = comgra_root_path / group
        self.trial_path = self.group_path / 'trials' / trial_id
        self.trial_path.mkdir(parents=True, exist_ok=True)
        self.configuration_type = None
        self.configuration_path = None
        self.prefixes_for_grouping_module_parameters = list(prefixes_for_grouping_module_parameters)
        assert all(isinstance(a, str) for a in prefixes_for_grouping_module_parameters)
        self.parameters_of_trial = parameters_of_trial
        self.set_of_top_level_modules = {}
        self.module_to_name = {}
        self.unique_module_names = {}
        self.unique_parameter_names = {}
        self.parameter_to_representation = {}
        self.decision_maker_for_recordings: DecisionMakerForRecordings = decision_maker_for_recordings
        self.current_stage = 'inactive'
        self.types_of_tensor_recordings = ['forward']
        self.current_type_of_tensor_recording = None
        self.tensor_recordings = TensorRecordings()
        #
        # Things that are recorded once and then compared to
        #
        self.configuration_type_to_status_and_graph: Dict[str, StatusAndGraph] = {}
        #
        # Per iteration
        #
        self.is_training_mode = False
        self.training_step = None
        self.iteration = None
        self.record_all_tensors_per_batch_index_by_default = False
        self.computation_step_to_tensor = {}
        self.tensor_to_name_and_iteration = {}
        self.tensor_name_and_iteration_to_representation: Dict[Tuple[str, int], TensorRepresentation] = {}
        self.current_batch_size = None

    @property
    def recording_is_active(self):
        return self.comgra_is_active and self.is_training_mode and self.decision_maker_for_recordings.is_record_on_this_iteration(self.training_step)

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

    def track_module(self, module_name, module: torch_nn.Module):
        self._track_module_recursive(module_name, module, self.set_of_top_level_modules, [])

    def _track_module_recursive(self, module_name, module: torch_nn.Module, container, preceding_names: List[str]):
        assert module_name not in container
        assert '.' not in module_name
        assert module not in self.module_to_name
        self.module_to_name[module] = module_name
        full_unique_name = '.'.join(preceding_names + [module_name])
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

    def get_name_of_module(self, module):
        return self.module_to_name[module]

    def start_next_recording(
            self, training_step, current_batch_size, is_training_mode,
            record_all_tensors_per_batch_index_by_default=False,
    ):
        self.is_training_mode = is_training_mode
        self.training_step = training_step
        self.iteration = 0
        self.record_all_tensors_per_batch_index_by_default = record_all_tensors_per_batch_index_by_default
        assert self.current_stage == 'inactive', self.current_stage
        self.current_stage = 'started'
        self.types_of_tensor_recordings = []
        self.current_type_of_tensor_recording = 'forward'
        self.computation_step_to_tensor = {}
        self.tensor_to_name_and_iteration = {}
        self.tensor_name_and_iteration_to_representation = {}
        self.current_batch_size = current_batch_size

    def register_tensor(
            self, tensor_name, tensor: torch.Tensor,
            index_of_batch_dimension=0,
            is_input=False, is_parameter=False, is_output=False, is_target=False, is_loss=False,
            recording_type=None, record_per_batch_index=None,
    ):
        if not self.recording_is_active:
            return
        assert (1 if is_input else 0) + (1 if is_parameter else 0) + (1 if is_output else 0) + \
               (1 if is_target else 0) + (1 if is_loss else 0) <= 1, tensor_name
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
            role = 'input'
        elif is_parameter:
            role = 'parameter'
        elif is_output:
            role = 'output'
        elif is_target:
            role = 'target'
        elif is_loss:
            role = 'loss'
        else:
            role = 'intermediate'
        self.computation_step_to_tensor[tensor.grad_fn] = tensor
        assert tensor not in self.tensor_to_name_and_iteration, \
            f"Tensor is already registered under the name and iteration {self.tensor_to_name_and_iteration[tensor]}"
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
            iteration=self.iteration,
            configuration_type=self.configuration_type,
            role=role,
            shape=list(tensor.shape),
            index_of_batch_dimension=index_of_batch_dimension,
            value_dimensions=list(value_dimensions),
            recording_type=recording_type,
            items_to_record=list(items_to_record),
            record_per_batch_index=record_per_batch_index,
        )
        self.tensor_name_and_iteration_to_representation[(tensor_name, self.iteration)] = tensor_representation
        # Store the current value of the tensor
        self.store_value_of_tensor(tensor, tensor_representation, self.iteration)

    def store_value_of_tensor(self, tensor: torch.Tensor, tensor_representation: TensorRepresentation, iteration):
        assert iteration == tensor_representation.iteration or tensor_representation.role == 'parameter'
        tensor_name = tensor_representation.full_unique_name
        value_dimensions = tensor_representation.value_dimensions
        if tensor_representation.recording_type == 'single_value':
            self.store_value_of_tensor_helper(
                'batch', iteration,
                tensor_name, 'single_value', None, tensor
            )
        else:
            batch_indices = ['batch'] + (list(range(self.current_batch_size)) if tensor_representation.record_per_batch_index else [])
            assert len(value_dimensions) > 0, tensor_name
            items_with_metadata = []
            for item in tensor_representation.items_to_record:
                if item == 'neurons':
                    assert len(value_dimensions) == 1, \
                        "This is not implemented yet for more than 1 dimension. To do so, use a view to combine " \
                        "all dimensions except the batch dimension."
                    for metadata in range(tensor.shape[value_dimensions[0]]):
                        items_with_metadata.append((item, metadata))
                else:
                    items_with_metadata.append((item, None))
            for item, metadata in items_with_metadata:
                # Aggregate over the value dimension, or extract the value at a given index of the value dimension
                if item == 'mean':
                    val = tensor.mean(dim=value_dimensions)
                elif item == 'abs_mean':
                    val = tensor.abs().mean(dim=value_dimensions)
                elif item == 'std':
                    val = tensor.std(dim=value_dimensions)
                elif item == 'abs_max':
                    val = torch.amax(tensor.abs(), dim=value_dimensions)
                elif item == 'neurons':
                    assert len(value_dimensions) == 1
                    if value_dimensions[0] == 0:
                        val = tensor[metadata, :]
                    elif value_dimensions[0] == 1:
                        val = tensor[:, metadata]
                    else:
                        assert False, \
                            f"With the current implementation of this, " \
                            f"tensors are assumed to have only two dimensions, batch and value."
                else:
                    raise NotImplementedError(item)
                # Take the mean over the batch, if possible
                for batch_index in batch_indices:
                    if tensor_representation.index_of_batch_dimension is None:
                        assert val.shape == (), (tensor_name, item, val.shape)
                        assert batch_index == 'batch'
                        val_specific_to_batch_index = val
                    else:
                        assert val.shape == (self.current_batch_size,), (tensor_name, item, val.shape)
                        assert tensor_representation.index_of_batch_dimension is not None or batch_index == 'batch'
                        if batch_index == 'batch':
                            val_specific_to_batch_index = val.mean()
                        else:
                            val_specific_to_batch_index = val[batch_index]
                    self.store_value_of_tensor_helper(
                        batch_index, iteration,
                        tensor_name, item, metadata, val_specific_to_batch_index
                    )

    def store_value_of_tensor_helper(self, batch_index, iteration, tensor_name, item, metadata, tensor):
        type_of_recording_to_batch_index_to_iteration_to_records = self.tensor_recordings.training_step_to_type_of_recording_to_batch_index_to_iteration_to_records.setdefault(
            self.training_step, {})
        batch_index_to_iteration_to_records = type_of_recording_to_batch_index_to_iteration_to_records.setdefault(
            self.current_type_of_tensor_recording, {})
        iteration_to_records = batch_index_to_iteration_to_records.setdefault(batch_index, {})
        records = iteration_to_records.setdefault(iteration, {})
        key = (tensor_name, item, metadata)
        assert key not in records, \
            f"Duplicate tensor recording for {self.training_step}, " \
            f"{self.current_type_of_tensor_recording}, {batch_index}, {key}"
        assert tensor.shape == (), (tensor.shape, key)
        records[key] = tensor

    def start_forward_pass(self, iteration, configuration_type):
        assert self.current_stage in ['started', 'after_iteration'], self.current_stage
        self.current_stage = 'forward'
        self.iteration = iteration
        assert isinstance(configuration_type, str) and re.match(r'[a-zA-Z_-]+', configuration_type), configuration_type
        self.configuration_type = configuration_type
        self.configuration_path = self.group_path / 'configs' / configuration_type
        self.configuration_path.mkdir(parents=True, exist_ok=True)
        self.tensor_recordings.iteration_to_configuration_type[self.iteration] = configuration_type
        if not self.recording_is_active:
            return

    def start_backward_pass(self):
        assert self.current_stage == 'forward', self.current_stage
        self.current_stage = 'backward'
        if not self.recording_is_active:
            return

    def record_current_gradients(self, name_of_loss_group):
        assert name_of_loss_group not in self.types_of_tensor_recordings
        self.types_of_tensor_recordings.append(name_of_loss_group)
        self.current_type_of_tensor_recording = name_of_loss_group
        for tensor, k in self.tensor_to_name_and_iteration.items():
            tr = self.tensor_name_and_iteration_to_representation[k]
            assert tensor.grad is not None, \
                f"A tensor does not have a gradient on it to record: {k}"
            self.store_value_of_tensor(tensor.grad, tr, self.iteration)

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
                    self.register_tensor(self.parameter_to_representation[t].full_unique_name, t, is_parameter=True)
            if t is not None:
                k = self.tensor_to_name_and_iteration[t]
                name_of_this_tensor, iteration = k
                tensor_representation = self.tensor_name_and_iteration_to_representation[k]
                assert iteration == self.iteration or tensor_representation.role == 'parameter', \
                    (name_of_this_tensor, iteration, self.iteration)
                if last_encountered_named_tensor_and_iteration is not None and \
                        last_encountered_named_tensor_and_iteration[0] not in tensor_representation.is_a_dependency_of:
                    dependency_is_from_same_iteration = (tensor_representation.iteration == last_encountered_named_tensor_and_iteration[1])
                    if dependency_is_from_same_iteration:
                        tensor_representation.is_a_dependency_of.append(last_encountered_named_tensor_and_iteration[0])
                assert tensor_representation.iteration is not None
                last_encountered_named_tensor_and_iteration = (name_of_this_tensor, tensor_representation.iteration)
                if tensor_representation.role == 'input':
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
            if self.tensor_name_and_iteration_to_representation[ni].role in ['output', 'target', 'loss']
            and self.tensor_name_and_iteration_to_representation[ni].iteration == self.iteration
        ]
        for tensor in final_tensors:
            traverse_graph_backwards(tensor.grad_fn, None)
        if self.configuration_type not in self.configuration_type_to_status_and_graph:
            assert sum([len(a.is_a_dependency_of) for a in self.tensor_name_and_iteration_to_representation.values()]) > 0, \
                "No computational graph could be constructed. " \
                "The most common error that could cause this is that gradient computations are turned off."
            #
            # Construct global status and graph information
            #
            status_and_graph = self.build_global_status_and_graph()
            # Make sure the graph and global_status are only DERIVED once per run
            self.configuration_type_to_status_and_graph[self.configuration_type] = status_and_graph
            # Make sure the graph and global_status are only SAVED once per set of multiple trials.
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
            new_version = self.build_global_status_and_graph()
            path = self.configuration_path / 'status_and_graph.pkl'
            with open(path, 'rb') as f:
                existing_version: StatusAndGraph = pickle.load(f)
            assert new_version.configuration_type == existing_version.configuration_type
            assert len(new_version.name_to_tensor_representation) == len(existing_version.name_to_tensor_representation)
            for k1, v1 in new_version.name_to_tensor_representation.items():
                v2 = dataclasses.asdict(existing_version.name_to_tensor_representation[k1])
                for kk1, vv1 in dataclasses.asdict(v1).items():
                    vv2 = v2[kk1]
                    vv1 = tuple(vv1) if isinstance(vv1, list) else vv1
                    vv2 = tuple(vv2) if isinstance(vv2, list) else vv2
                    assert vv1 == vv2
            assert tuple(new_version.types_of_tensor_recordings) == tuple(existing_version.types_of_tensor_recordings)
            assert tuple(new_version.get_all_items_to_record()) == tuple(existing_version.get_all_items_to_record()), \
                f"\n{new_version.get_all_items_to_record()}\n{existing_version.get_all_items_to_record()}"
            assert len(new_version.dag_format) == len(existing_version.dag_format)
            for a, b in zip(new_version.dag_format, existing_version.dag_format):
                assert len(a) == len(b)
                for c, d in zip(a, b):
                    assert c == d

    def build_global_status_and_graph(self) -> StatusAndGraph:
        nodes = [
            name for (name, iteration), v in self.tensor_name_and_iteration_to_representation.items()
            if iteration == self.iteration or v.role == 'parameter'
        ]
        connections = [
            [dependency, dependent]
            for (dependency, iteration), v in self.tensor_name_and_iteration_to_representation.items()
            for dependent in v.is_a_dependency_of
            if iteration == self.iteration or v.role == 'parameter'
        ]
        status_and_graph = StatusAndGraph(
            configuration_type=self.configuration_type,
            prefixes_for_grouping_module_parameters=list(self.prefixes_for_grouping_module_parameters),
            name_to_tensor_representation={
                name: v.create_copy_without_iteration_specific_data()
                for (name, iteration), v in self.tensor_name_and_iteration_to_representation.items()
                if iteration == self.iteration or v.role == 'parameter'
            },
            types_of_tensor_recordings=list(self.types_of_tensor_recordings),
            nodes=nodes,
            connections=connections,
        )
        status_and_graph.build_dag_format()
        return status_and_graph

    def finish_batch(self):
        assert self.current_stage == 'after_iteration', self.current_stage
        self.current_stage = 'inactive'
        if not self.recording_is_active:
            return
        #
        # Convert the TensorRecordings from tensor to float.
        # While doing so, minimize GPU-to-CPU transfers
        #
        self.decision_maker_for_recordings.prune_recordings(
            training_step=self.training_step, tensor_recordings=self.tensor_recordings
        )
        type_of_recording_to_batch_index_to_iteration_to_records = self.tensor_recordings.training_step_to_type_of_recording_to_batch_index_to_iteration_to_records[self.training_step]
        all_tensors = []
        for batch_index_to_iteration_to_records in type_of_recording_to_batch_index_to_iteration_to_records.values():
            for iteration_to_records in batch_index_to_iteration_to_records.values():
                for records in iteration_to_records.values():
                    for tensor in records.values():
                        all_tensors.append(tensor)
        combined_tensor = torch.stack(all_tensors)
        list_of_floats = combined_tensor.cpu().tolist()
        all_valid_keys = set(self.configuration_type_to_status_and_graph[self.configuration_type].get_all_items_to_record())
        c = 0
        for batch_index_to_iteration_to_records in type_of_recording_to_batch_index_to_iteration_to_records.values():
            for iteration_to_records in batch_index_to_iteration_to_records.values():
                for records in iteration_to_records.values():
                    for key in list(records.keys()):
                        assert key in all_valid_keys, key
                        records[key] = list_of_floats[c]
                        c += 1
        assert c == len(list_of_floats)
        with open(self.trial_path / 'recordings.pkl', 'wb') as f:
            pickle.dump(self.tensor_recordings, f)
