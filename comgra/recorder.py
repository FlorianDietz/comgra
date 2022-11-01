import dataclasses
import json
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch import nn as torch_nn

from comgra.objects import DirectedAcyclicGraph, GlobalStatus, ModuleRepresentation, ParameterRepresentation, TensorRepresentation


class ComgraRecorder:

    def __init__(
            self, comgra_root_path, group, trial_id, prefixes_for_grouping_module_parameters, parameters_of_trial
    ):
        comgra_root_path = Path(comgra_root_path)
        assert comgra_root_path.exists()
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
        self.current_stage = 'inactive'
        self.types_of_tensor_recordings = ['forward']
        self.current_type_of_tensor_recording = None
        #
        # Things that are recorded once and then compared to
        #
        self.global_status: Optional[GlobalStatus] = None
        self.graph: Optional[DirectedAcyclicGraph] = None
        #
        # Per iteration
        #
        self.recording_is_active = False
        self.computation_step_to_tensor = {}
        self.tensor_to_name = {}
        self.tensor_name_to_representation: Dict[str, TensorRepresentation] = {}

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

    def start_next_recording(self, recording_is_active=True):
        self.recording_is_active = recording_is_active
        assert self.current_stage == 'inactive', self.current_stage
        self.current_stage = 'started'
        self.types_of_tensor_recordings = []
        self.current_type_of_tensor_recording = 'forward'
        self.computation_step_to_tensor = {}
        self.tensor_to_name = {}
        self.tensor_name_to_representation = {}

    def register_tensor(
            self, tensor_name, tensor: torch.Tensor,
            batch_index=0,
            is_input=False, is_parameter=False, is_output=False, is_target=False, is_loss=False,
            recording_type='kpis',
    ):
        if not self.recording_is_active:
            return
        assert (1 if is_input else 0) + (1 if is_parameter else 0) + (1 if is_output else 0) + \
               (1 if is_target else 0) + (1 if is_loss else 0) <= 1, tensor_name
        if is_input:
            role = 'input'
            if not tensor.requires_grad:
                self._log_warning_once(
                    f"The input tensor {tensor_name} did not require a gradient, so comgra set requires_grad=True "
                    f"because this is needed to record the computational graph. This may have side effects."
                )
            tensor.requires_grad = True
        elif is_parameter:
            role = 'parameter'
        elif is_output:
            role = 'output'
        elif is_target:
            role = 'target'
            if not tensor.requires_grad:
                self._log_warning_once(
                    f"The target tensor {tensor_name} did not require a gradient, so comgra set requires_grad=True "
                    f"because this is needed to record the computational graph. This may have side effects."
                )
            tensor.requires_grad = True
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
            items_to_record = ['mean', 'std']
        elif recording_type == 'neurons':
            items_to_record = ['mean', 'std', 'neurons']
        else:
            raise NotImplementedError(recording_type)
        self.tensor_name_to_representation[tensor_name] = TensorRepresentation(
            full_unique_name=tensor_name,
            role=role,
            shape=list(tensor.shape),
            batch_index=batch_index,
            items_to_record=items_to_record,
        )

    def start_forward_pass(self):
        assert self.current_stage == 'started', self.current_stage
        self.current_stage = 'forward'
        if not self.recording_is_active:
            return
        #
        # Construct the computational graph
        #
        pass

    def start_backward_pass(self, name_of_loss_group):
        assert self.current_stage == 'forward', self.current_stage
        self.current_stage = 'backward'
        assert name_of_loss_group not in self.types_of_tensor_recordings
        self.types_of_tensor_recordings.append(name_of_loss_group)
        self.current_type_of_tensor_recording = name_of_loss_group
        if not self.recording_is_active:
            return
        # TODO this should be optional since there won't always be a loss.
        #  Also, there can be several losses and it may be good to track them separately.
        pass

    def finish_iteration(self):
        assert self.current_stage in ['forward', 'backward'], self.current_stage
        self.current_stage = 'inactive'
        self.current_type_of_tensor_recording = None
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
        # TODO
        #
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
