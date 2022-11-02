import dataclasses
import json
from pathlib import Path
import pickle
from typing import List, Dict, Optional

import torch
from torch import nn as torch_nn

from comgra.objects import DecisionMakerForRecordings, DirectedAcyclicGraph, GlobalStatus, ModuleRepresentation, ParameterRepresentation, TensorRecordings, TensorRepresentation


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
        self.prefixes_for_grouping_module_parameters = list(prefixes_for_grouping_module_parameters)
        assert all(isinstance(a, str) for a in prefixes_for_grouping_module_parameters)
        self.parameters_of_trial = parameters_of_trial
        self.trial_path.mkdir(parents=True, exist_ok=True)
        self._warning_messages_cache = {}
        self.set_of_top_level_modules = {}
        self.module_to_name = {}
        self.computational_graph_layout_and_global_data_have_been_recorded = False
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
        self.global_status: Optional[GlobalStatus] = None
        self.graph: Optional[DirectedAcyclicGraph] = None
        #
        # Per iteration
        #
        self.is_training_mode = False
        self.training_time = None
        self.record_all_tensors_per_batch_index_by_default = False
        self.computation_step_to_tensor = {}
        self.tensor_to_name = {}
        self.tensor_name_to_representation: Dict[str, TensorRepresentation] = {}
        self.current_batch_size = None

    @property
    def recording_is_active(self):
        return self.comgra_is_active and self.is_training_mode and self.decision_maker_for_recordings.is_record_on_this_iteration(self.training_time)

    def _log_warning_once(self, msg):
        if msg not in self._warning_messages_cache:
            print(msg)
            self._warning_messages_cache[msg] = True

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
            self, training_time, current_batch_size, is_training_mode,
            record_all_tensors_per_batch_index_by_default=False,
    ):
        self.is_training_mode = is_training_mode
        self.training_time = training_time
        self.record_all_tensors_per_batch_index_by_default = record_all_tensors_per_batch_index_by_default
        assert self.current_stage == 'inactive', self.current_stage
        self.current_stage = 'started'
        self.types_of_tensor_recordings = []
        self.current_type_of_tensor_recording = 'forward'
        self.computation_step_to_tensor = {}
        self.tensor_to_name = {}
        self.tensor_name_to_representation = {}
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
        if record_per_batch_index is None:
            record_per_batch_index = self.record_all_tensors_per_batch_index_by_default
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
        assert tensor not in self.tensor_to_name
        self.tensor_to_name[tensor] = tensor_name
        assert tensor_name not in self.tensor_name_to_representation, \
            f"Two tensors were recorded with the same name. Give your tensors unique names: {tensor_name}"
        if recording_type == 'kpis':
            items_to_record = ['mean', 'abs_mean', 'std']
        elif recording_type == 'neurons':
            items_to_record = ['mean', 'abs_mean', 'std', 'neurons']
        elif recording_type == 'single_value':
            items_to_record = ['single_value']
        else:
            raise NotImplementedError(recording_type)
        tensor_representation = TensorRepresentation(
            full_unique_name=tensor_name,
            role=role,
            shape=list(tensor.shape),
            index_of_batch_dimension=index_of_batch_dimension,
            value_dimensions=list(value_dimensions),
            recording_type=recording_type,
            items_to_record=list(items_to_record),
            record_per_batch_index=record_per_batch_index,
        )
        self.tensor_name_to_representation[tensor_name] = tensor_representation
        # Store the current value of the tensor
        self.store_value_of_tensor(tensor, tensor_representation)

    def store_value_of_tensor(self, tensor: torch.Tensor, tensor_representation: TensorRepresentation):
        tensor_name = tensor_representation.full_unique_name
        value_dimensions = tensor_representation.value_dimensions
        if tensor_representation.recording_type == 'single_value':
            self.store_value_of_tensor_helper(None, tensor_name, 'single_value', None, tensor)
        else:
            batch_indices = [None] + (list(range(self.current_batch_size)) if tensor_representation.record_per_batch_index else [])
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
                        assert batch_index is None
                        val_specific_to_batch_index = val
                    else:
                        assert val.shape == (self.current_batch_size,), (tensor_name, item, val.shape)
                        assert tensor_representation.index_of_batch_dimension is not None or batch_index is None
                        if batch_index is None:
                            val_specific_to_batch_index = val.mean()
                        else:
                            val_specific_to_batch_index = val[batch_index]
                    self.store_value_of_tensor_helper(batch_index, tensor_name, item, metadata, val_specific_to_batch_index)

    def store_value_of_tensor_helper(self, batch_index, tensor_name, item, metadata, tensor):
        type_of_recording_to_batch_index_to_records = self.tensor_recordings.training_time_to_type_of_recording_to_batch_index_to_records.setdefault(
            self.training_time, {})
        batch_index_to_records = type_of_recording_to_batch_index_to_records.setdefault(
            self.current_type_of_tensor_recording, {})
        records = batch_index_to_records.setdefault(batch_index, {})
        key = (tensor_name, item, metadata)
        assert key not in records, \
            f"Duplicate tensor recording for {self.training_time}, " \
            f"{self.current_type_of_tensor_recording}, {batch_index}, {key}"
        assert tensor.shape == ()
        records[key] = tensor

    def start_forward_pass(self):
        assert self.current_stage == 'started', self.current_stage
        self.current_stage = 'forward'
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
        for tensor, name in self.tensor_to_name.items():
            tr = self.tensor_name_to_representation[name]
            assert tensor.grad is not None, \
                f"A tensor does not have a gradient on it to record: {name}"
            self.store_value_of_tensor(tensor.grad, tr)

    def finish_iteration(self):
        assert self.current_stage in ['forward', 'backward'], self.current_stage
        self.current_stage = 'inactive'
        self.current_type_of_tensor_recording = 'forward'  # This will be used when the parameters get recorded in traverse_graph_backwards
        if not self.recording_is_active:
            return
        #
        # Go backwards through the computation graph, starting from outputs, targets, and losses.
        # Go back until you encounter an input, or you can't go back anymore.
        #
        def traverse_graph_backwards(step, last_encountered_named_tensor):
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
                if t in self.parameter_to_representation and t not in self.tensor_to_name:
                    self.register_tensor(self.parameter_to_representation[t].full_unique_name, t, is_parameter=True)
            if t is not None:
                name_of_this_tensor = self.tensor_to_name[t]
                tensor_representation = self.tensor_name_to_representation[name_of_this_tensor]
                if last_encountered_named_tensor is not None and \
                        last_encountered_named_tensor not in tensor_representation.is_a_dependency_of:
                    tensor_representation.is_a_dependency_of.append(last_encountered_named_tensor)
                last_encountered_named_tensor = name_of_this_tensor
                if tensor_representation.role == 'input':
                    return  # Do not track the graph beyond the inputs, which might go into the previous iteration.
            for predecessor, other in step.next_functions:
                traverse_graph_backwards(predecessor, last_encountered_named_tensor)
        final_tensors = [t for t, n in self.tensor_to_name.items() if self.tensor_name_to_representation[n].role in ['output', 'target', 'loss']]
        for tensor in final_tensors:
            traverse_graph_backwards(tensor.grad_fn, None)
        if not self.computational_graph_layout_and_global_data_have_been_recorded:
            assert sum([len(a.is_a_dependency_of) for a in self.tensor_name_to_representation.values()]) > 0, \
                "No computational graph could be constructed. " \
                "The most common error that could cause this is that gradient computations are turned off."
            print("-----------")
            for k, v in self.tensor_name_to_representation.items():
                print(k)
                print(v)
            #
            # Save global status information
            #
            assert self.global_status is None
            self.global_status = GlobalStatus(
                prefixes_for_grouping_module_parameters=list(self.prefixes_for_grouping_module_parameters),
                tensor_representations=dict(self.tensor_name_to_representation),
                types_of_tensor_recordings=list(self.types_of_tensor_recordings),
            )
            with open(self.group_path / 'globals.json', 'w') as f:
                json.dump(dataclasses.asdict(self.global_status), f)
            #
            # Construct a graph format and save it.
            #
            nodes = list(self.tensor_name_to_representation.keys())
            connections = [
                [dependency, dependent]
                for dependency, rep in self.tensor_name_to_representation.items()
                for dependent in rep.is_a_dependency_of
            ]
            assert self.graph is None
            self.graph = DirectedAcyclicGraph(
                nodes=nodes,
                connections=connections,
            )
            self.graph.build_dag_format(self.global_status)
            with open(self.group_path / 'graph.json', 'w') as f:
                json.dump(dataclasses.asdict(self.graph), f)
            # Make sure this is only done once
            self.computational_graph_layout_and_global_data_have_been_recorded = True
        #
        # Save trial information
        #
        with open(self.trial_path / 'parameters.json', 'w') as f:
            json.dump(self.parameters_of_trial, f)
        #
        # Verify that the result is identical to previous results.
        # For the graph.
        #
        nodes = list(self.tensor_name_to_representation.keys())
        connections = [
            [dependency, dependent]
            for dependency, rep in self.tensor_name_to_representation.items()
            for dependent in rep.is_a_dependency_of
        ]
        fake_graph = DirectedAcyclicGraph(
            nodes=nodes,
            connections=connections,
        )
        fake_graph.build_dag_format(self.global_status)
        assert len(fake_graph.dag_format) == len(self.graph.dag_format)
        for a, b in zip(fake_graph.dag_format, self.graph.dag_format):
            assert len(a) == len(b)
            for c, d in zip(a, b):
                assert c == d
        #
        # Verify that the result is identical to previous results.
        # For the global_data.
        #
        assert tuple(self.types_of_tensor_recordings) == tuple(self.global_status.types_of_tensor_recordings)
        fake_global_status = GlobalStatus(
            prefixes_for_grouping_module_parameters=list(self.prefixes_for_grouping_module_parameters),
            tensor_representations=dict(self.tensor_name_to_representation),
            types_of_tensor_recordings=list(self.types_of_tensor_recordings),
        )
        assert tuple(self.global_status.get_all_items_to_record()) == tuple(fake_global_status.get_all_items_to_record()), \
            f"\n{self.global_status.get_all_items_to_record()}\n{fake_global_status.get_all_items_to_record()}"
        #
        # Convert the TensorRecordings from tensor to float.
        # While doing so, minimize GPU-to-CPU transfers
        #
        type_of_recording_to_batch_index_to_records = self.tensor_recordings.training_time_to_type_of_recording_to_batch_index_to_records[self.training_time]
        all_tensors = []
        for batch_index_to_records in type_of_recording_to_batch_index_to_records.values():
            for records in batch_index_to_records.values():
                for tensor in records.values():
                    all_tensors.append(tensor)
        combined_tensor = torch.stack(all_tensors)
        list_of_floats = combined_tensor.cpu().tolist()
        all_valid_keys = set(self.global_status.get_all_items_to_record())
        c = 0
        for batch_index_to_records in type_of_recording_to_batch_index_to_records.values():
            for records in batch_index_to_records.values():
                for key in list(records.keys()):
                    assert key in all_valid_keys, key
                    records[key] = list_of_floats[c]
                    c += 1
        assert c == len(list_of_floats)
        with open(self.trial_path / 'recordings.json', 'wb') as f:
            pickle.dump(self.tensor_recordings, f)
