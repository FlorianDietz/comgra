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
from typing import List, Dict, Optional, Tuple, Set, Union, Any

import msgpack

import torch
from torch import nn as torch_nn

import comgra
from comgra.objects import SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS, \
    DecisionMakerForRecordings, TrainingStepConfiguration, NodeGraphStructure, ModuleRepresentation, Node, \
    ParameterRepresentation, TensorRecordings, TensorReference, TensorRepresentation, GraphConfigurationOfOneIteration, \
    TensorGraphStructure
from comgra import utilities


class ComgraRecorder:

    def __init__(
            self, comgra_root_path, group, trial_id,
            decision_maker_for_recordings,
            max_num_batch_size_to_record=None,
            prefixes_for_grouping_module_parameters_visually=None,
            prefixes_for_grouping_module_parameters_in_nodes=None,
            comgra_is_active=True,
            max_num_mappings_to_save_at_once_during_serialization=10000,
            type_of_serialization='msgpack',
            calculate_svd_and_other_expensive_operations_of_parameters=True,
    ):
        """
        The main entry point for recording data with comgra.

        :param comgra_root_path: The path where comgra will store all the data it extracts.
        :param group: The name of a folder that will be created within comgra_root_path and combines several trials. When you start the comgra server, the path you give to the server should include this folder. All trials written to the same folder will be visualized in parallel in the GUI.
        :param trial_id: The name for this trial run. If you want to compare multiple independent trials in the comgra GUI, give them unique names here and save them with the same 'group' parameter.
        :param decision_maker_for_recordings: An object that determines how often comgra makes a recording.
        :param max_num_batch_size_to_record: The number of indices of each batch that get recorded in detail. If you set this too high, it may take up a lot of memory and space on the hard drive. Set this to None to record all values.
        :param prefixes_for_grouping_module_parameters_visually: Use this to group similar module parameters visually into the same column in the dependency graph. All module parameters whose complete name (including the list of names of modules they are contained in) match one of these prefixes are grouped together.
        :param prefixes_for_grouping_module_parameters_in_nodes: Building on prefixes_for_grouping_module_parameters_visually, group the parameters together into a single node.
        :param comgra_is_active: Set this to False if you want to turn this library off.
        :param max_num_mappings_to_save_at_once_during_serialization: An optional parameter you can experiment with if comgra is too slow. If this is too low, comgra becomes slow. If this is too high, the program may crash due to memory problems. (This problem is caused by a serialization bug in a backend library.)
        :param type_of_serialization: Several options for serialization exist. It is recommended to keep the default.
        :param calculate_svd_and_other_expensive_operations_of_parameters: An optional feature to record statistics that are more expensive to calculate than others.
        """
        comgra_root_path = Path(comgra_root_path).resolve()
        assert comgra_root_path.exists(), comgra_root_path
        self.comgra_is_active = comgra_is_active
        self.trial_id = trial_id
        self.group_path = comgra_root_path / group
        self.trial_path = self.group_path / 'trials' / trial_id
        self.recordings_path = self.trial_path / 'recordings'
        self.configurations_path = self.trial_path / 'configurations'
        if comgra_is_active:
            shutil.rmtree(self.trial_path, ignore_errors=True)
            self.recordings_path.mkdir(parents=True, exist_ok=True)
            self.configurations_path.mkdir(parents=True, exist_ok=True)
        self.type_of_serialization = type_of_serialization
        self.calculate_svd_and_other_expensive_operations_of_parameters = calculate_svd_and_other_expensive_operations_of_parameters
        self.prefixes_for_grouping_module_parameters_visually = list(prefixes_for_grouping_module_parameters_visually or [''])
        self.prefixes_for_grouping_module_parameters_in_nodes = list(prefixes_for_grouping_module_parameters_in_nodes or [])
        assert all(isinstance(a, str) for a in self.prefixes_for_grouping_module_parameters_visually)
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
        self.most_recent_training_step_where_execution_type_was_recorded = collections.defaultdict(lambda: -1)
        self.all_different_types_of_execution_have_been_encountered = False
        #
        # KPI graph recording
        #
        self.kpi_graph_exponential_backoff_factor = 1.1
        self.kpi_graph_history_to_check_for_outliers = 6
        self.kpi_graph_factor_for_detecting_outliers = 0.5
        self.kpi_graph_excerpt = {}
        self.kpi_graph_changed = False
        self.kpi_graph_next_training_step_to_update_file = 10.0
        #
        # Things that are recorded once and then compared to
        #
        self.cache_hash_to_node_graph_structure: Dict[str, NodeGraphStructure] = {}
        #
        # Per training_step
        #
        self.training_step_configuration: Optional[TrainingStepConfiguration] = None
        self.batch_indices_categories_and_string_representations_to_record: Optional[List[Tuple[int, str]]] = None
        self.tensor_recordings: Optional[TensorRecordings] = None
        self.mapping_of_tensors_for_extracting_kpis: Dict[Tuple[int, str, Optional[int], Optional[int], str, str, str], Tuple[torch.Tensor, TensorRepresentation]] = {}
        self.training_step = None
        self.type_of_execution = None
        self.list_of_delayed_function_calls = []
        self.iteration = None
        self.record_all_tensors_per_batch_index_by_default = False
        self.tensor_to_list_of_references: Dict[torch.Tensor, List[TensorReference]] = {}  # The first of these tuples is the canonical one
        self.tensors_that_were_manually_marked_to_require_grad_but_dont_need_to_be_recorded: Set[torch.Tensor] = set()
        self.canonical_tensor_reference_to_tensor: Dict[TensorReference, torch.Tensor] = {}
        self.tensor_reference_to_representation: Dict[TensorReference, TensorRepresentation] = {}
        self.manual_tensor_connections_sink_to_sources_by_tensor: Dict[torch.Tensor, List[torch.Tensor]] = {}
        self.manual_tensor_connections_sink_to_sources_by_computation_step: Dict[Any, List[torch.Tensor]] = {}
        self.current_batch_size = None
        self.override__recording_is_active = None
        self.sanity_check__recursion_for_delayed_calls = None

    def recording_is_active(self):
        """
        :return: True if comgra is currently recording, False otherwise.
        """
        if self.override__recording_is_active is None:
            if not self.comgra_is_active:
                return False
            if self.type_of_execution is None:
                # While the type_of_execution is unknown (it will be set later by decide_recording_of_batch()),
                # pretend that the recording is active.
                return True
            return self.decision_maker_for_recordings.is_record_on_this_step(
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
        """
        Register a module (and all of its submodules recursively) so that comgra keeps track of its parameters.
        :param module_name: The name with which the module will show up in the GUI
        :param module: A pytorch module
        """
        if not self.comgra_is_active:
            return
        assert self.current_stage == 'inactive', "Modules should be tracked before starting any recordings."
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
    def add_note(self, note):
        """
        This function acts as a simple logger, in case you want to view text messages.
        The notes can be viewed in their own tab in the GUI.
        These messages are always logged, independent of whether a recording is made on this training step.
        :param note: A string you want to log.
        """
        if not self.comgra_is_active:
            return
        self.notes.append(str(note))
        with open(self.trial_path / 'notes.json', 'w') as f:
            json.dump(self.notes, f)

    @utilities.runtime_analysis_decorator
    def get_name_of_module(self, module):
        """
        Get the name of a module that you registered with :py:func:`~comgra.recorder.ComgraRecorder.track_module`. This works recursively. The names of contained modules are constructed by adding the field name to the name of the parent.
        :param module: A pytorch module.
        :return: The name of the module, as specified by :py:func:`~comgra.recorder.ComgraRecorder.track_module`.
        """
        return self.module_to_name[module]

    @utilities.runtime_analysis_decorator
    def start_batch(
            self, training_step, current_batch_size,
            type_of_execution='main_execution_type',
            record_all_tensors_per_batch_index_by_default=True,
            override__recording_is_active=None,
    ):
        """
        Start a new recording for a new training step. Each recording can consist of multiple iterations.
        :param training_step: An integer that represents the current training step.
        :param current_batch_size: An integer that represents the current batch size.
        :param type_of_execution: An optional string that assigns this recording to a category.
        The GUI allows you to filter by this category.
        If this is set to None, it is treated as uncertain whether recordings should be made until :py:func:`~comgra.recorder.ComgraRecorder.decide_recording_of_batch` is called.
        This allows for dynamic recordings to be made.
        However, be aware that it also slows down your code: Some operations need to be done immediately and can't be delayed until after the decision to record is made.
        This leads to a slowdown. The severity of the slowdown depends on your network. In one of our more complex examples, the experiments took up to 20% longer.
        :param record_all_tensors_per_batch_index_by_default: If True,
        :py:func:`~comgra.recorder.ComgraRecorder.register_tensor` will act as if record_per_batch_index was True by default.
        :param override__recording_is_active: Override decision_maker_for_recordings of :py:obj:`~comgra.recorder.ComgraRecorder`.
        None = use decision_maker_for_recordings to decide whether to record (default)
        True = record
        False = don't record
        Important: Set this to False when you are running the network in evaluation mode, as torch will not generate any computation graphs in this case, so comgra can't work.
        :return:
        """
        assert self.training_step is None or training_step > self.training_step, \
            f"The training_step should increase monotonically."
        self.training_step = training_step
        if type_of_execution is not None:
            assert type_of_execution != 'any_value', "Don't use 'any_value', it has a special meaning in the GUI."
            assert not type_of_execution.startswith('__'), type_of_execution
            assert re.match(r'^[a-zA-Z0-9-_]+$', type_of_execution)
        self.type_of_execution = type_of_execution
        self.iteration = None
        self.record_all_tensors_per_batch_index_by_default = record_all_tensors_per_batch_index_by_default
        assert self.current_stage == 'inactive', self.current_stage
        self.current_stage = 'started'
        self.current_type_of_tensor_recording = 'forward'
        self.current_batch_size = current_batch_size
        self._reset_caches()
        self.sanity_check__recursion_for_delayed_calls = False
        # If type_of_execution is None, we normally have to wait until the user decides whether to record or not later.
        # But if all_different_types_of_execution_have_been_encountered then we can also decide early
        if self.all_different_types_of_execution_have_been_encountered:
            if override__recording_is_active is None:
                if not any(self.decision_maker_for_recordings.is_record_on_this_step(self.training_step, toe)
                           for toe, step in self.most_recent_training_step_where_execution_type_was_recorded.items()):
                    override__recording_is_active = False
        self.override__recording_is_active = override__recording_is_active
        if not self.recording_is_active():
            return
        assert self.set_of_top_level_modules, \
            "No modules have been defined yet. Use track_module() on your modules before starting a recording."
        #
        # Note:
        # The below code is run immediately, even if type_of_execution is None.
        # In this case we need to wait for decide_recording_of_batch() before running code and should wrap code
        # in helper functions that we pass to _run_now_or_add_to_delayed_calls().
        # However, this is already done by register_tensor() internally
        #
        self.training_step_configuration = TrainingStepConfiguration(
            type_of_execution=self.type_of_execution,  # This may be None, but must be set later, before serialization
            modules_and_parameters=self.set_of_top_level_modules,
        )
        # Register the tensors on the parameters
        for tensor, parameter in self.parameter_to_representation.items():
            tensor_name = parameter.full_unique_name
            node_name = next(
                (a for a in self.prefixes_for_grouping_module_parameters_in_nodes if tensor_name.startswith(a)), None)
            self.register_tensor(tensor_name, tensor, is_parameter=True, node_name=node_name)

    def _reset_caches(self):
        # Note that this method should be called both before and after each training step.
        # It empties caches, which allows pytorch to free memory.
        self.list_of_delayed_function_calls = []
        self.training_step_configuration = None
        self.batch_indices_categories_and_string_representations_to_record = None
        self.tensor_recordings = TensorRecordings()
        self.mapping_of_tensors_for_extracting_kpis = {}
        self.types_of_tensor_recordings = []
        self.tensor_to_list_of_references = {}
        self.tensors_that_were_manually_marked_to_require_grad_but_dont_need_to_be_recorded = set()
        self.canonical_tensor_reference_to_tensor = {}
        self.tensor_reference_to_representation = {}
        self.manual_tensor_connections_sink_to_sources_by_tensor = collections.defaultdict(list)
        self.manual_tensor_connections_sink_to_sources_by_computation_step = collections.defaultdict(list)

    @utilities.runtime_analysis_decorator
    def register_tensor(
            self, tensor_name, tensor: torch.Tensor,
            index_of_batch_dimension=0,
            is_input=False, is_parameter=False, is_target=False, is_loss=False,
            recording_type=None, record_per_batch_index=None,
            node_name=None, role_within_node=None, is_initial_value=False,
    ):
        """
        Register a tensor. Each tensor registered in this way will either become its own node in the GUI, or is assigned to part of a node.

        This function will set .requires_grad in order to make sure that the tensor is tracked by the computation graph,
        even if it e.g. is an input and does not normally require a gradient.

        Because of the way requires_grad works, this needs to happen before the tensor is used in another computation,
        or else the attribute will have no effect. Therefore, register_tensor() should be called on a tensor before
        that tensor is used in another computation.

        :param tensor_name: The name with which the tensor should be registered. The name should be unique per iteration.
        :param tensor: A pytorch tensor.
        :param index_of_batch_dimension: The index of the batch dimension of the tensor.
        :param is_input: Specify that the tensor should be considered an input of the network.
        :param is_parameter: Specify that the tensor should be considered an input of the network. Endusers should not need this option: Use :py:func:`~comgra.recorder.ComgraRecorder.track_module` instead.
        :param is_target: Specify that the tensor should be considered a target of the network.
        :param is_loss: Specify that the tensor should be considered a loss value of the network.
        :param recording_type: A categorical value that indicates what kind of data of the tensor should be extracted and recorded.
        :param record_per_batch_index: If True, a record is made for each sample of the batch instead of just summary statistics. Only uses as many samples as specified with :py:obj:`~comgra.recorder.ComgraRecorder.max_num_batch_size_to_record`
        :param node_name: An optional string that assigns this tensor to a node. All tensors with the same node_name will share a node in the GUI.
        :param role_within_node: If a node_name is specified, give this tensor a role within the node to differentiate it from the other tensors in the node.
        :param is_initial_value: Use is_initial_value=True to register the initial value of a hidden state after calling :py:func:`~comgra.recorder.ComgraRecorder.start_batch` but before :py:func:`~comgra.recorder.ComgraRecorder.start_iteration`.
        :return: None
        """
        if not self.recording_is_active():
            return
        # Make sure that gradients are generated and retained for later.
        if not tensor.requires_grad:
            tensor.requires_grad = True
        tensor.retain_grad()

        def function_to_run():
            nonlocal node_name, role_within_node, index_of_batch_dimension, recording_type, record_per_batch_index
            assert self.iteration is not None or is_initial_value or is_parameter, \
                "You must call start_iteration() before registering any tensors, unless you use is_initial_value=True."
            assert not is_initial_value or self.iteration is None, \
                ("You can only use is_initial_value=True when registering a tensor "
                 "before the first call of start_iteration().")
            assert not is_initial_value or self.current_stage == 'started', \
                ("Use is_initial_value=True to register the initial value of a hidden state after "
                 "calling start_batch() but before start_iteration().")
            iteration_to_use_for_registration = -1 if is_initial_value else self.iteration
            assert (1 if is_input else 0) + (1 if is_parameter else 0) + \
                   (1 if is_target else 0) + (1 if is_loss else 0) <= 1, tensor_name
            assert not tensor_name.startswith('node__')
            node_name = 'node__' + (tensor_name if node_name is None else node_name)
            role_within_node = tensor_name if role_within_node is None else role_within_node
            # Safety checks to avoid overlap with SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS
            assert '*' not in tensor_name, "tensor_name can't contain an asterisk."
            assert '*' not in node_name, "node_name can't contain an asterisk."
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
                    recording_type = 'neuron'
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
            # Create a TensorReference for the tensor and store various references for later.
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
            if recording_type == 'kpis':
                items_to_record = ['mean', 'abs_mean', 'std', 'abs_max']
            elif recording_type == 'kpis_and_svd':
                items_to_record = ['mean', 'abs_mean', 'std', 'abs_max', 'svd']
            elif recording_type == 'neuron':
                items_to_record = ['mean', 'abs_mean', 'std', 'abs_max', 'neuron']
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
                    type_of_tensor=type_of_tensor,
                    shape=list(tensor.shape),
                    index_of_batch_dimension=index_of_batch_dimension,
                    value_dimensions=list(value_dimensions),
                    recording_type=recording_type,
                    items_to_record=list(items_to_record),
                    record_per_batch_index=record_per_batch_index,
                )
                self.tensor_reference_to_representation[tensor_reference] = tensor_representation
                self.canonical_tensor_reference_to_tensor[tensor_reference] = tensor
                # Store the current value of the tensor
                self._store_value_of_tensor(tensor, tensor_representation)
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
        self._run_now_or_add_to_delayed_calls(function_to_run)

    @utilities.runtime_analysis_decorator
    def add_tensor_connection(
            self, src: Union[torch.Tensor, str, TensorReference], sink: Union[torch.Tensor, str, TensorReference]
    ):
        """
        Create a connection in the dependency graph, from the src tensor to the sink tensor.
        :param src: The start of the connection, as either a tensor, the name of a registered tensor, or a :py:obj:`~comgra.objects.TensorReference`.
        :param sink: The end of the connection, as either a tensor, the name of a registered tensor, or a :py:obj:`~comgra.objects.TensorReference`.
        :return:
        """
        if not self.recording_is_active():
            return

        def function_to_run():
            nonlocal src, sink
            def verify_object(obj, identifier):
                if isinstance(obj, str):
                    matching_references = []
                    for tensor, refs in self.tensor_to_list_of_references.items():
                        for ref in refs:
                            if ref.tensor_name == obj:
                                matching_references.append((tensor, ref))
                    assert matching_references, f"There is no tensor matching the name '{obj}' for the {identifier}."
                    # If the name matches multiple registered tensors, pick the one with the highest iteration
                    max_iteration = max(ref.iteration for _, ref in matching_references)
                    matching_references = [(t, ref) for t, ref in matching_references if ref.iteration == max_iteration]
                    assert len(matching_references) == 1, \
                        (f"Programming error. This error should never be visible to users. "
                         f"Two tensors where registered with the same name for the same iteration. "
                         f"register_tensor() should raise an exception when this happens.\n"
                         f"{identifier}\n{obj}\n{max_iteration}\n{matching_references}")
                    obj = matching_references[0][0]
                if isinstance(obj, TensorReference):
                    obj = self.canonical_tensor_reference_to_tensor[obj.get_canonical_reference()]
                assert isinstance(obj, torch.Tensor), \
                    f"The {identifier} must be either a tensor, the name of a registered tensor, or a TensorReference."
                assert obj.requires_grad, \
                    (f"The {identifier} must have the requires_grad attribute set to True.\n"
                     f"Without this, backpropagating through the computation graph may not arrive at this tensor "
                     f"(in case of sinks) or propagate past it (in case of sources).\n"
                     f"Two additional notes:\n"
                     f"(1) comgra's register_tensor() function will set .requires_grad automatically.\n"
                     f"(2) You need to set .requires_grad immediately, before using the tensor in another computation, "
                     f"or else the attribute won't have an effect. "
                     f"This also means that you should call register_tensor() immediately after defining a tensor.")
                return obj
            src = verify_object(src, 'src')
            sink = verify_object(sink, 'sink')
            assert src is not sink
            # Store the sink in up to two different ways, depending on whether grad_fn is set or not,
            # because this determines how the sink tensor can be discovered later:
            # Either by recursing through the computation graph's grad_fn elements, or as a leaf tensor.
            # Note that if the sink is stored both ways then the graph backpropagation algorithm may end up recursing
            # through it twice. It should be able to handle that without issue, just like it has to be able to handle
            # the possibility that a grad_fn leads to the same tensor twice by two different routes.
            if sink.grad_fn is not None:
                # We compare by id() and set() because pytorch will try to compare objects with == on lists
                if id(src) not in set(id(a) for a in self.manual_tensor_connections_sink_to_sources_by_computation_step[sink.grad_fn]):
                    self.manual_tensor_connections_sink_to_sources_by_computation_step[sink.grad_fn].append(src)
            # We compare by id() and set() because pytorch will try to compare objects with == on lists
            if id(src) not in set(id(a) for a in self.manual_tensor_connections_sink_to_sources_by_tensor[sink]):
                self.manual_tensor_connections_sink_to_sources_by_tensor[sink].append(src)
        self._run_now_or_add_to_delayed_calls(function_to_run)

    @utilities.runtime_analysis_decorator
    def detach_while_keeping_connection(self, tensor: torch.Tensor):
        """
        Return tensor.detach(), but use add_tensor_connection() to connect the detached tensor to the original,
        ensuring that the connection still shows up in the dependency graph.
        """
        res = tensor.detach()
        res.requires_grad = True
        self.tensors_that_were_manually_marked_to_require_grad_but_dont_need_to_be_recorded.add(res)
        self.add_tensor_connection(tensor, res)
        return res

    @utilities.runtime_analysis_decorator
    def _store_value_of_tensor(self, tensor: torch.Tensor, tensor_representation: TensorRepresentation):
        tensor_name = tensor_representation.original_reference.tensor_name
        value_dimensions = tensor_representation.value_dimensions
        if tensor_representation.recording_type == 'single_value':
            self._store_value_of_tensor_helper(
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
                    val = self._get_highest_svd(tensor)
                elif item == 'neuron':
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
                        max_num_batch_size_to_record = self.current_batch_size if self.max_num_batch_size_to_record is None else min(
                            self.current_batch_size, self.max_num_batch_size_to_record)
                        assert len(val.shape) == 2 and val.shape[0] == self.current_batch_size, \
                            (val.shape, self.current_batch_size, max_num_batch_size_to_record)
                        if self.batch_indices_categories_and_string_representations_to_record is None:
                            # If no specific samples were requested, just get the first ones
                            self.batch_indices_categories_and_string_representations_to_record = [(i, "", f"Sample {i}") for i in range(max_num_batch_size_to_record)]
                        assert 0 < len(self.batch_indices_categories_and_string_representations_to_record) <= max_num_batch_size_to_record
                        # If the selected samples happen to be the first ones, select them directly for increased performance
                        if len(set(category for _, category, _ in self.batch_indices_categories_and_string_representations_to_record)) == 1 \
                                and min(batch_index for batch_index, _, _ in self.batch_indices_categories_and_string_representations_to_record) == 0 \
                                and max(batch_index for batch_index, _, _ in self.batch_indices_categories_and_string_representations_to_record) == max_num_batch_size_to_record - 1:
                            val1 = val[0:max_num_batch_size_to_record, :]
                        else:
                            indices_to_pick = [batch_index for batch_index, _, _ in self.batch_indices_categories_and_string_representations_to_record]
                            val1 = val[indices_to_pick, :]
                        assert val1.shape == (len(self.batch_indices_categories_and_string_representations_to_record), val.shape[1]), \
                            (val1.shape, (len(self.batch_indices_categories_and_string_representations_to_record), val.shape[1]))
                    else:
                        assert False, batching_type
                    self._store_value_of_tensor_helper(
                        batching_type, tensor_representation.original_reference.iteration,
                        tensor_representation, item, val1,
                    )

    @utilities.runtime_analysis_decorator
    def _get_highest_svd(self, tensor):
        try:
            res = torch.linalg.svdvals(tensor)[:1]
        except:
            res = torch.tensor([float('nan')], device=tensor.device)
        return res

    @utilities.runtime_analysis_decorator
    def _store_value_of_tensor_helper(self, batching_type, iteration, tensor_representation: TensorRepresentation, item, tensor):
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
    def start_iteration(self):
        """
        Tell comgra that a new iteration has started. Should be called after :py:func:`~comgra.recorder.ComgraRecorder.start_batch`.
        """
        assert self.current_stage in ['started', 'after_iteration'], self.current_stage
        self.current_stage = 'forward'
        if not self.recording_is_active():
            return
        def function_to_run():
            self.iteration = 0 if self.iteration is None else (self.iteration + 1)
        self._run_now_or_add_to_delayed_calls(function_to_run)

    @utilities.runtime_analysis_decorator
    def _run_now_or_add_to_delayed_calls(self, lam, always_run_this=False):
        """
        If type_of_execution was provided when :py:func:`~comgra.recorder.ComgraRecorder.start_batch()`
        was called, (i.e. if self.type_of_execution is not None)
        then execute the provided function immediately.
        Else store it in a list for later.
        When :py:func:`~comgra.recorder.ComgraRecorder.decide_recording_of_batch`
        is called it will either discard or execute these functions.
        :param lam:
        :param always_run_this: If True, the lambda is always run even if :py:func:`~comgra.recorder.ComgraRecorder.recording_is_active` is False after :py:func:`~comgra.recorder.ComgraRecorder.decide_recording_of_batch` is called.
        :return:
        """
        assert self.recording_is_active() or always_run_this, "This should have been verified before this function is called"
        assert not self.sanity_check__recursion_for_delayed_calls, \
            "Programming error. Functions used with _run_now_or_add_to_delayed_calls() should not be able to call each other."
        if self.type_of_execution is None:
            self.list_of_delayed_function_calls.append((lam, always_run_this))
        else:
            self.sanity_check__recursion_for_delayed_calls = True
            lam()
            self.sanity_check__recursion_for_delayed_calls = False

    def declare_that_all_different_types_of_execution_have_been_encountered(self):
        """
        A helper function that can mitigate the slowdown caused by :py:func:`~comgra.recorder.ComgraRecorder.decide_recording_of_batch`.
        Calling this function tells comgra that no new type_of_execution values will be encountered in the future.
        (If this turns out to be wrong, it causes an error)
        This allows comgra to speed things up a bit because it can now skip all pre-recordings that happen before
        :py:func:`~comgra.recorder.ComgraRecorder.decide_recording_of_batch` if all the type_of_recording values
        seen so far have recently been recorded.
        """
        self.all_different_types_of_execution_have_been_encountered = True

    @utilities.runtime_analysis_decorator
    def decide_recording_of_batch(self, type_of_execution: str, category_per_sample: List[str]):
        """
        This function is needed if :py:func:`~comgra.recorder.ComgraRecorder.start_batch` was called with type_of_execution=None.
        (Else it does nothing and returns immediately.)
        It sets the missing value for type_of_execution and if that type_of_execution should be recorded
        on this training step, then it runs all functions that have been delayed so far.
        It also selects which indices of the batch should be recorded:
        Each sample in the batch may be assigned a category.
        Comgra will try to record an equal number of samples from each category.
        :param type_of_execution: The type_of_execution.
        :param category_per_sample: A list of strings that determine the category of a sample. Each item corresponds to one sample in the batch.
        :return: None
        """
        if self.type_of_execution is None:
            assert type_of_execution is not None
            assert type_of_execution != 'any_value', "Don't use 'any_value', it has a special meaning in the GUI."
            assert not type_of_execution.startswith('__'), type_of_execution
            assert re.match(r'^[a-zA-Z0-9-_]+$', type_of_execution)
            self.type_of_execution = type_of_execution
            if self.training_step_configuration is not None:
                self.training_step_configuration.type_of_execution = type_of_execution
        else:
            assert self.type_of_execution == type_of_execution, \
                (f"start_batch() was called with a different type_of_execution than decide_recording_of_batch()\n"
                 f"{self.type_of_execution}, {type_of_execution}")
            assert not self.list_of_delayed_function_calls, \
                ("start_batch() was already called with a type_of_execution, so functions should have been "
                 "executed immediately.")
            # Nothing to do in this case, because all functions should have already been executed.
            return
        if not self.recording_is_active():
            self._execute_functions_in_list_of_delayed_function_calls(only_functions_that_mustnt_be_skipped=True)
            return
        # If we get here then we should make a recording on this training_step.
        # First, define which samples to record: Try to get an equal number from every category
        assert len(category_per_sample) == self.current_batch_size, \
            (len(category_per_sample), self.current_batch_size)
        assert all(isinstance(category, str) for category in category_per_sample), category_per_sample
        category_to_batch_indices = collections.defaultdict(list)
        for i, category in enumerate(category_per_sample):
            category_to_batch_indices[category].append(i)
        max_num_samples_for_any_category = max(len(a) for a in category_to_batch_indices.values())
        self.batch_indices_categories_and_string_representations_to_record = []
        i = 0
        batch_size_to_record = self.current_batch_size if self.max_num_batch_size_to_record is None else min(
            self.current_batch_size, self.max_num_batch_size_to_record)
        digits_needed_for_printing_any_batch_size = len(str(self.current_batch_size - 1))
        while len(self.batch_indices_categories_and_string_representations_to_record) < batch_size_to_record and i < max_num_samples_for_any_category:
            for category, batch_indices in category_to_batch_indices.items():
                if i < len(batch_indices) and len(self.batch_indices_categories_and_string_representations_to_record) < batch_size_to_record:
                    batch_index = batch_indices[i]
                    string_representation = f"Sample {batch_index:0{digits_needed_for_printing_any_batch_size}d} | {category}"
                    self.batch_indices_categories_and_string_representations_to_record.append(
                        (batch_index, category, string_representation)
                    )
            i += 1
        assert 0 < len(self.batch_indices_categories_and_string_representations_to_record) <= batch_size_to_record, \
            (len(self.batch_indices_categories_and_string_representations_to_record), batch_size_to_record)
        self.batch_indices_categories_and_string_representations_to_record.sort(key=lambda a: (a[1], a[0]))
        self._execute_functions_in_list_of_delayed_function_calls(only_functions_that_mustnt_be_skipped=False)

    def _execute_functions_in_list_of_delayed_function_calls(self, only_functions_that_mustnt_be_skipped=False):
        # Execute all functions that were stored in list_of_delayed_function_calls.
        self.sanity_check__recursion_for_delayed_calls = True
        for lam, always_run_this in self.list_of_delayed_function_calls:
            if not only_functions_that_mustnt_be_skipped or always_run_this:
                lam()
        self.sanity_check__recursion_for_delayed_calls = False
        self.list_of_delayed_function_calls = None

    @utilities.runtime_analysis_decorator
    def record_current_gradients(self, name_of_loss_group, set_gradients_to_zero_if_not_a_parameter=False):
        """
        Tell comgra to save all gradients that are currently on all registered tensors and parameters. Should be called after :py:func:`~comgra.recorder.ComgraRecorder.start_iteration` but before :py:func:`~comgra.recorder.ComgraRecorder.finish_iteration`.
        @param name_of_loss_group: The gradients are stored under this name. It is possible to call this function multiple times with different names in order to save multiple different sets of gradients.
        """
        if not self.recording_is_active():
            return
        if self.type_of_execution is None:
            raise ValueError(
                "If you set type_of_execution=None in start_batch(), "
                "decide_recording_of_batch() should be called before record_current_gradients(). "
                "This is necessary because you may assign multiple gradients to a tensor if you use multiple loss "
                "functions, and this will result in incorrect recordings of the gradients if "
                "record_current_gradients() has to delay its execution."
            )
        assert name_of_loss_group not in self.types_of_tensor_recordings, \
            (name_of_loss_group, self.types_of_tensor_recordings)
        self.types_of_tensor_recordings.append(name_of_loss_group)
        self.current_type_of_tensor_recording = name_of_loss_group
        for tensor, refs in self.tensor_to_list_of_references.items():
            tr = self.tensor_reference_to_representation[refs[0]]
            gradient = torch.zeros(tensor.shape, device=tensor.device) if tensor.grad is None else tensor.grad
            self._store_value_of_tensor(gradient, tr)
            if set_gradients_to_zero_if_not_a_parameter and tr.type_of_tensor != 'parameter':
                tensor.grad = None

    @utilities.runtime_analysis_decorator
    def finish_iteration(self):
        """
        Tell comgra that the iteration has ended. Should be called after :py:func:`~comgra.recorder.ComgraRecorder.start_iteration` but before :py:func:`~comgra.recorder.ComgraRecorder.finish_batch`.
        """
        assert self.current_stage == 'forward', self.current_stage
        self.current_stage = 'after_iteration'
        if not self.recording_is_active():
            return
        def function_to_run():
            self._finish_iteration()
        self._run_now_or_add_to_delayed_calls(function_to_run)

    @utilities.runtime_analysis_decorator
    def _finish_iteration(self):
        self.current_type_of_tensor_recording = 'forward'  # This will be used when the parameters get recorded in traverse_graph_backwards
        #
        # Go backwards through the computation graph, starting from outputs, targets, and losses.
        # Go back until you encounter an input, or you can't go back anymore.
        #
        computation_step_to_tensor = {
            tensor.grad_fn: tensor for tensor
            in self.tensor_to_list_of_references.keys()
            if tensor.grad_fn is not None
        }
        assert None not in computation_step_to_tensor, computation_step_to_tensor[None]
        assert len(set(computation_step_to_tensor.values())) == len(computation_step_to_tensor)
        assert len(set(computation_step_to_tensor.keys())) == len(computation_step_to_tensor)
        for k, v in computation_step_to_tensor.items():
            assert k is v.grad_fn, (self.iteration, self.tensor_to_list_of_references[v][0].tensor_name)
        cache_to_avoid_duplicate_calls__tensor_references = set()
        cache_to_avoid_duplicate_calls__computation_graph = set()
        tensor_references_to_use_for_this_iteration = set()
        tensor_reference_to_list_of_dependents: Dict[TensorReference, List[TensorReference]] = collections.defaultdict(list)

        @utilities.runtime_analysis_decorator
        def traverse_graph_backwards__tensor(
                tensor: torch.Tensor,
                previously_encountered_tensor_references: List[Optional[TensorReference]],
                previously_followed_manual_connections: List[Tuple[int, int]],
                this_was_called_because_of_a_manual_connection=False,
        ):
            """
            This function sets dependencies between tensors.
            It is called once for each tensor that definitely should show up in this iteration because the user
            registered it, and it also backpropagates and includes other tensors along the way.
            During backpropagation along the computation graph, a tensor may be encountered more than once
            (since each tensor may have multiple dependents) but should only be processed once.
            ---
            Note the following problem with backpropagation:
            You can't go from the computation graph (steps_to_follow) to the tensors directly,
            unless you manually registered the connection beforehand, as is done here with computation_step_to_tensor.
            You can only go the other way around, from the tensor to the graph, using .grad_fn,
            but unfortunately not all tensors actually have a grad_fn attribute.
            This is why it is necessary to use two separate helper functions here:
            traverse_graph_backwards__tensor() and traverse_graph_backwards__computation_graph()
            ---
            Design goals for special cases:
            If a tensor is registered multiple times,
            the references that belong to this iteration are ordered in a chain.
            Dependencies of the tensor connect to the leftmost item of the chain,
            dependents connect to the rightmost item.
            If the tensor is registered multiple times, but for an older iteration only, then only the most recent
            of the references to those iterations is used as a dependency and the others are ignored
            (In terms of implementation, a new reference is created that copies the most recent older reference).
            ---
            Other special cases to consider:
            If a tensor has no grad_fn, you can't backpropagate through it and you find it as a leaf variable in the graph.
            If it does have a grad_fn, it can't be found directly during backpropagation and computation_step_to_tensor has to be used instead.
            If a tensor has no requires_grad, it won't be found during backpropagation.
            To cover all these cases, it is necessary for the code to discover tensors both by recursing through the graph
            and by manually visiting tensors directly, through add_tensor_connection(),
            manual_tensor_connections_sink_to_sources_by_tensor and manual_tensor_connections_sink_to_sources_by_computation_step
            """
            last_encountered_reference = previously_encountered_tensor_references[-1] if previously_encountered_tensor_references else None
            # Skip duplicate calls.
            key = (tensor, last_encountered_reference)
            if key in cache_to_avoid_duplicate_calls__tensor_references:
                return
            cache_to_avoid_duplicate_calls__tensor_references.add(key)
            #
            # The following sections registers dependencies based on the tensor
            # and creates additional references if required.
            # It may be called multiple times, once per last_encountered_reference,
            # and this should not cause issues.
            # It updates previously_encountered_tensor_references as a side effect, which is used when recursing.
            #
            keep_recursing = True
            assert (tensor in self.tensor_to_list_of_references) or this_was_called_because_of_a_manual_connection
            if tensor in self.tensor_to_list_of_references:
                # Get the first and the last reference to this tensor on this iteration.
                # Or if there is no reference on this iteration yet, create a new reference.
                refs = self.tensor_to_list_of_references[tensor]
                tmp = [ref for ref in refs if ref.iteration == self.iteration]
                if not tmp:
                    # Create a new reference based on the last reference to this tensor that is not itself a duplicate.
                    # I.e., if the same tensor gets registered in several nodes by the user, this will pick whichever
                    # one of these was created last and use that as the canonical reference.
                    reference_to_copy = [
                        ref for ref in refs
                        if not ref.tensor_name.endswith(SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS)
                    ][-1]
                    if self.tensor_reference_to_representation[reference_to_copy.get_canonical_reference()].type_of_tensor == 'parameter':
                        # Parameters are a special case since they are only registered once,
                        # and that is done automatically, not by the user.
                        # There can be only one older (node, iteration, role_within_node) tuple for each parameter,
                        # so creating a new one with the current iteration will be unique even if the role_within_node
                        # is not changed.
                        reference_to_copy = refs[0]
                        assert reference_to_copy.iteration == -1
                        role_within_node_suffix = f''
                    elif reference_to_copy.iteration == -1:
                        role_within_node_suffix = '__from_initialization'
                    else:
                        role_within_node_suffix = f'__from_iteration_{reference_to_copy.iteration}'
                    self.tensor_to_list_of_references[tensor].append(TensorReference(
                        tensor_name=reference_to_copy.tensor_name + SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS,
                        iteration=self.iteration,
                        node_name=reference_to_copy.node_name + SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS,
                        # node_name=original_ref.node_name + f'__reuse_iteration_{original_ref.iteration}' + SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS,
                        role_within_node=reference_to_copy.role_within_node + role_within_node_suffix,
                        is_canonical_reference=False,
                        previous_reference=self.tensor_to_list_of_references[tensor][-1],
                    ))
                    refs = self.tensor_to_list_of_references[tensor]
                    tmp = [ref for ref in refs if ref.iteration == self.iteration]
                    assert len(tmp) == 1
                for ref in tmp:
                    tensor_references_to_use_for_this_iteration.add(ref)
                # Set dependencies between all references of this tensor that will be displayed at the same time
                for dependency_ref, dependent_ref in zip(tmp[:-1], tmp[1:]):
                    # This list should be empty the first time this code block is reached,
                    # and should be set to the same one-element list on all subsequent times.
                    assert len(tensor_reference_to_list_of_dependents[dependency_ref]) == 0 or \
                           tuple(tensor_reference_to_list_of_dependents[dependency_ref]) == (dependent_ref,), \
                        ("Programming error. "
                         "Dependencies between references of the same tensor should form an uninterrupted chain. "
                         "No part of the code except this one here should be able to add more references in between "
                         "them. All dependents of the tensor should attach to the rightmost reference, "
                         "all dependencies to the leftmost reference.")
                    tensor_reference_to_list_of_dependents[dependency_ref] = [dependent_ref]
                    assert dependency_ref is not dependent_ref
                first_ref = tmp[0]
                last_ref = tmp[-1]
                tensor_representation = self.tensor_reference_to_representation[refs[0]]
                # Add last_encountered_reference to the list of dependents of last_ref
                if last_encountered_reference is not None:
                    if last_ref is last_encountered_reference and previously_followed_manual_connections:
                        raise ValueError(
                            f"The tensor {last_ref.tensor_name} on iteration {last_ref.iteration} was "
                            f"encountered twice while recursing through the computation graph. "
                            f"A manual connection defined by add_tensor_connection() was encountered during this "
                            f"process and may be responsible. Please ensure that add_tensor_connection() does not "
                            f"introduce any cycles in your computation graph.")
                    assert last_ref is not last_encountered_reference, last_ref
                    assert last_encountered_reference not in tensor_reference_to_list_of_dependents[last_ref], \
                        ("Programming error. If a dependency is registered twice, that means the graph traversal "
                         "has some redundancy and could be optimized. "
                         "This should be prevented by cache_to_avoid_duplicate_calls.")
                    tensor_reference_to_list_of_dependents[last_ref].append(last_encountered_reference)
                # If this reference is an import of an earlier reference, do not recurse
                # NOTE: This used to be more restrictive. It used to also filter on type_of_tensor == 'input'.
                # I removed this in part because showing what contributed to inputs is not actually bad, and in part
                # because it was bugged and fixing it much harder than it seems.
                # The issue was that there can be multiple inputs depending on each other,
                # and type_of_tensor == 'input' caused those connections to be skipped.
                # If I decide to change this back later, keep in mind a special case that makes this harder:
                # old0->input0->input1, old0->new1.
                # A ll of those tensors should be shown, and the connection old0->input0
                # should then also appear in the graph and not get dropped.
                if first_ref.node_name.endswith(SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS):
                    keep_recursing = False
                # Set the previously_encountered_tensor_references
                previously_encountered_tensor_references = previously_encountered_tensor_references + list(reversed(tmp))
                assert previously_encountered_tensor_references[-1] is first_ref
            #
            # Recurse
            #
            if keep_recursing:
                # Recurse through the computation graph, accessed via .grad_fn
                if tensor.grad_fn is not None:
                    if not this_was_called_because_of_a_manual_connection:  # 'tensor' could not be registered in this case
                        assert computation_step_to_tensor[tensor.grad_fn] is tensor, \
                            (self.tensor_to_list_of_references[tensor][0],)
                    for predecessor, _ in tensor.grad_fn.next_functions:
                        if predecessor is not None:
                            traverse_graph_backwards__computation_graph(
                                predecessor, previously_encountered_tensor_references,
                                previously_followed_manual_connections,
                            )
                # Recurse through self.manual_tensor_connections_sink_to_sources_by_tensor
                helper_to_recurse_through_manual_tensor_connections(
                    tensor,
                    None,
                    previously_encountered_tensor_references,
                    previously_followed_manual_connections,
                )

        @utilities.runtime_analysis_decorator
        def traverse_graph_backwards__computation_graph(
                step_to_follow,
                previously_encountered_tensor_references: List[Optional[TensorReference]],
                previously_followed_manual_connections: List[Tuple[int, int]],
        ):
            """
            See documentation of traverse_graph_backwards__tensor().
            """
            assert step_to_follow is not None
            last_encountered_reference = previously_encountered_tensor_references[-1] if previously_encountered_tensor_references else None
            # Skip duplicate calls.
            # It's possible for this function to be called twice with the same arguments:
            # There could be two or more paths through the computation graph that go from last_encountered_reference
            # to the current tensor.
            key = (step_to_follow, last_encountered_reference)
            if key in cache_to_avoid_duplicate_calls__computation_graph:
                return
            cache_to_avoid_duplicate_calls__computation_graph.add(key)
            # Get the registered tensor, if there is one
            # (we may be at a computation step that lies in between two registered tensors)
            t = None
            if step_to_follow in computation_step_to_tensor:
                assert not hasattr(step_to_follow, 'variable'), \
                    "This shouldn't be possible. hasattr(step, 'variable') is True if it's a leaf, " \
                    "while computation_step_to_tensor is used for intermediate values."
                t = computation_step_to_tensor[step_to_follow]
            if hasattr(step_to_follow, 'variable'):
                assert (step_to_follow.variable in self.parameter_to_representation
                        or step_to_follow.variable in self.tensor_to_list_of_references
                        or step_to_follow.variable in self.tensors_that_were_manually_marked_to_require_grad_but_dont_need_to_be_recorded), \
                    ("Backpropagation encountered a leaf tensor that has not been registered. "
                     "This can happen if you set .requires_grad = True on a tensor that isn't a parameter "
                     "and isn't registered with register_tensor().")
                if (step_to_follow.variable in self.parameter_to_representation or
                        step_to_follow.variable in self.tensor_to_list_of_references):
                    t = step_to_follow.variable
            #
            # If this step created a tensor that is registered,
            # stop recursing and switch to traverse_graph_backwards__tensor.
            # Else continue recursing along the computation graph.
            #
            if t is not None:
                traverse_graph_backwards__tensor(
                    t, previously_encountered_tensor_references,
                    previously_followed_manual_connections,
                )
            else:
                for predecessor, _ in step_to_follow.next_functions:
                    if predecessor is not None:
                        traverse_graph_backwards__computation_graph(
                            predecessor, previously_encountered_tensor_references,
                            previously_followed_manual_connections,
                        )
                helper_to_recurse_through_manual_tensor_connections(
                    None,
                    step_to_follow,
                    previously_encountered_tensor_references,
                    previously_followed_manual_connections,
                )
                if hasattr(step_to_follow, 'variable') and isinstance(step_to_follow.variable, torch.Tensor):
                    helper_to_recurse_through_manual_tensor_connections(
                        step_to_follow.variable,
                        None,
                        previously_encountered_tensor_references,
                        previously_followed_manual_connections,
                    )
        def helper_to_recurse_through_manual_tensor_connections(
                sink_as_tensor,
                sink_as_computation_step,
                previously_encountered_tensor_references: List[Optional[TensorReference]],
                previously_followed_manual_connections: List[Tuple[int, int]],
        ):
            assert (sink_as_tensor is None) != (sink_as_computation_step is None), "One of the two options"
            if sink_as_tensor is not None:
                source_tensors = [
                    a for a in self.manual_tensor_connections_sink_to_sources_by_tensor.get(sink_as_tensor, [])
                ]
                sink_id = id(sink_as_tensor)
            else:
                source_tensors = [
                    a for a in self.manual_tensor_connections_sink_to_sources_by_computation_step.get(sink_as_computation_step, [])
                ]
                sink_id = id(sink_as_computation_step)
            for source_tensor in source_tensors:
                # Sanity check
                connection = (id(source_tensor), sink_id)
                if connection in previously_followed_manual_connections:
                    raise ValueError(
                        f"A loop was encountered while constructing the dependency graph in comgra. "
                        f"This was caused by add_tensor_connection(). "
                        f"Please make sure you did not accidentally use that function to create a loop. "
                        f"The following tensor references were involved in the loop (named tensors only!):\n"
                        f"{', '.join([str((a.tensor_name, a.iteration)) for a in previously_encountered_tensor_references if a is not None])}"
                    )
                traverse_graph_backwards__tensor(
                    source_tensor, previously_encountered_tensor_references,
                    previously_followed_manual_connections + [connection],
                    this_was_called_because_of_a_manual_connection=True,
                )
        #
        # Backpropagate from any tensor that was created or otherwise referenced in this iteration
        #
        tensors_to_show = [
            tensor for tensor, refs in self.tensor_to_list_of_references.items()
            for ref in refs
            if ref.iteration == self.iteration
        ]
        for tensor in tensors_to_show:
            traverse_graph_backwards__tensor(tensor,[], [])
        #
        # Sanity check
        #
        for ref1 in tensor_references_to_use_for_this_iteration:
            for ref2 in tensor_references_to_use_for_this_iteration:
                if ref1.get_canonical_reference() == ref2.get_canonical_reference():
                    assert ((ref1.iteration < self.iteration) and (ref1 == ref2)) or \
                           (ref1.iteration == self.iteration), \
                        ("Programming error. All references to the same tensor should belong to the same iteration. "
                         "Either multiple from this one, or one from a previous one. "
                         "If this assertion fails, then probably traverse_graph_backwards() has a bug")
        assert any([ref1 for ref1, refs in tensor_reference_to_list_of_dependents.items() if
                    any([ref2 for ref2 in refs if ref1.get_canonical_reference() != ref2.get_canonical_reference()])]), \
            "No computational graph could be constructed. " \
            "The most common error that could cause this is that gradient computations are turned off."
        assert not any((dependency in dependents) for dependency, dependents in tensor_reference_to_list_of_dependents.items())
        #
        # Build the dependency graph based on the data extracted while recursing through the computation graph
        #
        self._initialize_graph_structure_objects_at_end_of_iteration(
            tensor_references_to_use_for_this_iteration,
            tensor_reference_to_list_of_dependents,
        )

    @utilities.runtime_analysis_decorator
    def _initialize_graph_structure_objects_at_end_of_iteration(
            self,
            tensor_references_to_use_for_this_iteration: Set[TensorReference],
            tensor_reference_to_list_of_dependents: Dict[TensorReference, List[TensorReference]],
    ):
        #
        # Build the NodeGraphStructure
        #
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
                )
                name_to_node[node_name] = node
            assert tr.type_of_tensor == node.type_of_tensor, \
                f"Node {node_name} stores tensors with different type_of_tensor." \
                f"\n{tr.type_of_tensor}\n{node.type_of_tensor}"
        #
        # TensorGraphStructure
        #
        tensor_connections = [
            (dependency, dependent)
            for (dependency, dependents) in tensor_reference_to_list_of_dependents.items()
            for dependent in dependents
        ]
        assert len(tensor_connections) == len(set(tensor_connections)), \
            ("Programming error. This should have no duplicates. "
             "If it does, probably the graph traversal has redundant steps and could be optimized.")
        tensor_graph_structure = TensorGraphStructure(
            tensor_references=sorted(list(tensor_references_to_use_for_this_iteration), key=lambda a: a.tensor_name),
            tensor_connections=tensor_connections,
        )
        #
        # NodeGraphStructure
        #
        node_graph_structure = NodeGraphStructure(
            name_to_node=name_to_node,
        )
        node_graph_structure.build_dag_format(
            self, tensor_references_to_use_for_this_iteration, tensor_reference_to_list_of_dependents,
            self.tensor_reference_to_representation, tensor_connections,
        )
        # Compare the node_graph_structure with existing ones in the cache.
        if node_graph_structure.node_graph_hash in self.cache_hash_to_node_graph_structure:
            # If it wasn't new, run a simple sanity check,
            # asserting that it is in fact the same as the thing in the cache.
            try:
                utilities.recursive_equality_check(
                    self.cache_hash_to_node_graph_structure[node_graph_structure.node_graph_hash],
                    node_graph_structure,
                    []
                )
            except AssertionError as e:
                raise ValueError(
                    f"Two NodeGraphStructure objects received the same hash but they have different content."
                ) from e
            # Replace it by reference to save some memory
            node_graph_structure = self.cache_hash_to_node_graph_structure[node_graph_structure.node_graph_hash]
        else:
            # If it's new, cache it
            self.cache_hash_to_node_graph_structure[node_graph_structure.node_graph_hash] = node_graph_structure
            # Save it to file
            node_graph_structure_folder = self.group_path / 'node_graph_structure'
            node_graph_structure_file = node_graph_structure_folder / f'{node_graph_structure.node_graph_hash}.pkl'
            node_graph_structure_folder.mkdir(parents=True, exist_ok=True)
            # If a file for it already exists even though it is not in the cache, load it and compare.
            # This can happen if comgra is run multiple times without deleting previous trials
            if node_graph_structure_file.exists():
                try:
                    with open(node_graph_structure_file, 'rb') as f:
                        existing_version: NodeGraphStructure = pickle.load(f)
                    utilities.recursive_equality_check(existing_version, node_graph_structure, [])
                except AssertionError as e:
                    raise ValueError(
                        f"A file for the NodeGraphStructure with the same hash already exists "
                        f"but it contains different content: {node_graph_structure.node_graph_hash}"
                    ) from e
            else:
                with open(node_graph_structure_file, 'wb') as f:
                    pickle.dump(node_graph_structure, f)
        #
        # Update training step data
        #
        assert len(self.training_step_configuration.graph_configuration_per_iteration) == self.iteration
        assert node_graph_structure.node_graph_hash != 'TBD'
        self.training_step_configuration.graph_configuration_per_iteration.append(GraphConfigurationOfOneIteration(
            hash_of_node_graph_structure=node_graph_structure.node_graph_hash,
            tensor_graph_structure=tensor_graph_structure,
        ))

    @utilities.runtime_analysis_decorator
    def finish_batch(self):
        """
        Tell comgra that the recording has ended. Should be called after :py:func:`~comgra.recorder.ComgraRecorder.finish_iteration`.
        """
        assert self.current_stage == 'after_iteration' or not self.recording_is_active(), self.current_stage
        self.current_stage = 'inactive'
        if not self.recording_is_active():
            return
        if self.type_of_execution is None:
            raise ValueError(
                "If you set type_of_execution=None in start_batch(), "
                "decide_recording_of_batch() should be called before finish_batch()."
            )
        assert not self.list_of_delayed_function_calls, \
            "This list should be empty at this point because of decide_recording_of_batch()"
        # Make the recorder remember that we used this type_of_execution
        self.decision_maker_for_recordings.mark_recording_on_this_step(self.training_step, self.type_of_execution)
        # Save the TrainingStepConfiguration
        self._save_training_step_configuration()
        # Save the tensors
        self._save_tensor_recordings()
        # Save the graph of KPIs, which is independent of the rest of the recordings
        self._save_recorded_kpi_graphs_if_needed(False)
        # Update
        if self.all_different_types_of_execution_have_been_encountered:
            if self.type_of_execution not in self.most_recent_training_step_where_execution_type_was_recorded:
                raise ValueError(
                    f"declare_that_all_different_types_of_execution_have_been_encountered() was called "
                    f"too early. The type_of_execution '{self.type_of_execution}' was not encountered before."
                )
        self.most_recent_training_step_where_execution_type_was_recorded[self.type_of_execution] = self.training_step
        # Clear caches
        self._reset_caches()

    def finalize(self):
        """
        Save anything that is still in the buffer.
        """
        if not self.comgra_is_active:
            return
        # Save the graph of KPIs, which is independent of the rest of the recordings
        self._save_recorded_kpi_graphs_if_needed(True)
        # Clear caches
        self._reset_caches()

    def _save_training_step_configuration(self):
        training_step_configuration_path = self.configurations_path / f'{self.training_step}.pkl'
        assert not training_step_configuration_path.exists(), training_step_configuration_path
        with open(training_step_configuration_path, 'wb') as f:
            pickle.dump(self.training_step_configuration, f)

    def _save_tensor_recordings(self):
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
            self._save_file(dump_dict, recordings_path_folder, file_number)
            file_number += 1
            self.tensor_recordings.recordings = utilities.PseudoDb(attributes=attributes_for_tensor_recordings)

        batch_size_to_record = len(self.batch_indices_categories_and_string_representations_to_record) if \
            self.batch_indices_categories_and_string_representations_to_record is not None else \
            (self.current_batch_size if self.max_num_batch_size_to_record is None else min(
                self.current_batch_size, self.max_num_batch_size_to_record))
        attributes_for_tensor_recordings = [
            'training_step', 'type_of_tensor_recording', 'batch_aggregation', 'iteration',
            'node_name', 'role_within_node', 'record_type', 'item', 'metadata',
        ]
        self.tensor_recordings.recordings = utilities.PseudoDb(attributes=attributes_for_tensor_recordings)
        # Save a preliminary file with information about each tensor.
        # (This is saved in the database just in case the values differ between different iterations, etc.)
        # (For example, the shape of a tensor may in rare cases depend on the iteration.)
        assert len(self.tensor_reference_to_representation) == len(
            set([(tr.original_reference.tensor_name, tr.original_reference.iteration) for tr in
                 self.tensor_reference_to_representation.values()])), \
            (f"Programming error. TensorRepresentations are saved in duplicates, which may clog the database.\n"
             f"{len(self.tensor_reference_to_representation), len(set([(tr.original_reference.tensor_name, tr.original_reference.iteration) for tr in self.tensor_reference_to_representation.values()]))}")
        for main_ref, tr in self.tensor_reference_to_representation.items():
            for item, metadata, val in [
                ('tensor_shape', None, list(tr.shape)),
                ('index_of_batch_dimension', None, tr.index_of_batch_dimension),
            ]:
                final_key = self.training_step, 'not_applicable', 'not_applicable', main_ref.iteration, main_ref.node_name, main_ref.role_within_node, 'meta_information', item, metadata
                self._add_tensor_recordings_for_key_and_register_alternate_references(final_key, val, main_ref)
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
                batch_values = [
                    string_representation
                    for batch_index, category, string_representation
                    in self.batch_indices_categories_and_string_representations_to_record
                ]
                assert tensor.shape[0] == batch_size_to_record, (tensor.shape, self.current_batch_size, batch_size_to_record)
            else:
                batch_values = [batching_type]
                assert tensor.shape[0] == 1, (tensor.shape, self.current_batch_size, batch_size_to_record, key)
            if item == 'neuron':
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
            assert tensor.numel() == len(batch_values) * len(neuron_values), \
                (tensor.shape, len(batch_values), len(neuron_values))
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
                    self._add_tensor_recordings_for_key_and_register_alternate_references(
                        key_to_process, float_value, main_ref
                    )
                    sanity_check_c += 1
                all_tensors_to_combine = []
                all_keys_to_process = []
                save_recordings_so_far()
        assert len(all_tensors_to_combine) == 0
        total_number_of_tensor_values = sum(t.numel() for t, tr in self.mapping_of_tensors_for_extracting_kpis.values())
        assert sanity_check_c == total_number_of_tensor_values, (sanity_check_c, total_number_of_tensor_values)

    def _add_tensor_recordings_for_key_and_register_alternate_references(self, key, float_value, ref: TensorReference):
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
    def _save_file(self, dump_dict, recordings_path_folder, file_number):
        if self.type_of_serialization == 'json':
            return self._save_json(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'zip_json':
            return self._save_zip_json(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'pkl':
            return self._save_pickle(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'msgpack':
            return self._save_msgpack(dump_dict, recordings_path_folder, file_number)
        elif self.type_of_serialization == 'zip_msgpack':
            return self._save_zip_msgpack(dump_dict, recordings_path_folder, file_number)
        else:
            raise ValueError(self.type_of_serialization)

    @utilities.runtime_analysis_decorator
    def _save_json(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.json'
        with open(path, 'w') as f:
            json.dump(dump_dict, f)
        return path

    @utilities.runtime_analysis_decorator
    def _save_zip_json(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.zip_json'
        with gzip.open(path, 'w') as fout:
            json_bytes = (json.dumps(dump_dict) + "\n").encode('utf-8')
            fout.write(json_bytes)
        return path

    @utilities.runtime_analysis_decorator
    def _save_pickle(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.pkl'
        with open(path, 'wb') as f:
            pickle.dump(dump_dict, f)
        return path

    @utilities.runtime_analysis_decorator
    def _save_msgpack(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.msgpack'
        with open(path, 'wb') as f:
            msgpack.dump(dump_dict, f)
        return path

    @utilities.runtime_analysis_decorator
    def _save_zip_msgpack(self, dump_dict, recordings_path_folder, file_number):
        path = recordings_path_folder / f'{file_number}.zip_msgpack'
        with gzip.open(path, 'wb') as fout:
            fout.write(msgpack.dumps(dump_dict))
        return path

    @utilities.runtime_analysis_decorator
    def record_kpi_in_graph(
            self, kpi_group, kpi_name, val,
            timepoint=None, record_even_if_recording_is_inactive=False,
    ):
        """
        Create graphs, similar to tensorboard. These can be inspected in their own tab in the GUI. A separate graph is automatically created for each separate type_of_execution. You can also use the parameters of this function to create subgroups.

        The recording of graphs saves memory by using exponential falloff to determine when to save: It saves with a high frequency early on, then waits longer and longer. If an outlier is encountered, it ignores this rule and records the outlier anyway.

        :param kpi_group: The name of the graph that this value will be written to.
        :param kpi_name: The name of the line in the graph that this value will be written to.
        :param val: The value to store. Either a one-element tensor or a number.
        :param timepoint: The timepoint to use for the x-axis. Defaults to the training_step.
        :param record_even_if_recording_is_inactive: If True, graph values will be recorded even if comgra is not recording tensors on this training_step. This can be useful if you use your own conditions for when to call this function and those conditions rarely coincide with comgra being active. This argument is False by default because checking whether to record requires a GPU-to-CPU transfer, and we don't want to make these unnecessarily often.
        """
        if not self.comgra_is_active:
            return
        if not self.recording_is_active() and not record_even_if_recording_is_inactive:
            return
        def function_to_run():
            nonlocal timepoint, val
            if timepoint is None:
                timepoint = self.training_step
            assert self.type_of_execution is not None
            stats = self.kpi_graph_excerpt.setdefault(kpi_group, {}).setdefault(self.type_of_execution, {}).setdefault(kpi_name, {
                'vals': [],
                'next_timepoint': 0,
                'last_timepoint': -1,
            })
            if isinstance(val, torch.Tensor):
                # This is an expensive operation, so only do it if self.recording_is_active()
                # We don't want to move results from the GPU only to throw them away later in the function
                val = val.item()
            if len(stats['vals']) >= self.kpi_graph_history_to_check_for_outliers:
                history_to_check = [a['val'] for a in stats['vals'][-self.kpi_graph_history_to_check_for_outliers:]]
                max_ = max(history_to_check)
                min_ = min(history_to_check)
                dist = max_ - min_
                is_outlier = (val > max_ + dist * self.kpi_graph_factor_for_detecting_outliers
                              or val < min_ - dist * self.kpi_graph_factor_for_detecting_outliers)
            else:
                is_outlier = False
            assert timepoint > stats['last_timepoint'], f"Must be called with a newer timepoint each time."
            if timepoint >= stats['next_timepoint'] or is_outlier:
                assert isinstance(val, numbers.Number)
                stats['vals'].append({
                    'timepoint': timepoint,
                    'val': val,
                })
                while timepoint >= stats['next_timepoint']:
                    stats['next_timepoint'] = max([1, stats['next_timepoint'] * self.kpi_graph_exponential_backoff_factor])
                    stats['last_timepoint'] = timepoint
                self.kpi_graph_changed = True
        self._run_now_or_add_to_delayed_calls(function_to_run, always_run_this=record_even_if_recording_is_inactive)

    @utilities.runtime_analysis_decorator
    def _save_recorded_kpi_graphs_if_needed(self, finalize):
        if not finalize:
            if not self.kpi_graph_changed:
                return
            if self.training_step < self.kpi_graph_next_training_step_to_update_file:
                return
            self.kpi_graph_next_training_step_to_update_file *= self.kpi_graph_exponential_backoff_factor
        self.kpi_graph_changed = False
        # Save the file with a tmp suffix first, then overwrite the real one.
        # This prevents issues in case the visualizer is accessing the file while it is being overwritten.
        tmp_path = self._save_file(self.kpi_graph_excerpt, self.trial_path, 'kpi_graph_tmp')
        real_path = tmp_path.parent / f'kpi_graph{tmp_path.suffix}'
        os.replace(tmp_path, real_path)
