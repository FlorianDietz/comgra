# BASIC IMPORTS
from abc import ABC, abstractmethod
import base64
import collections
from collections import defaultdict, OrderedDict
import copy
from dataclasses import dataclass
import datetime
import functools
import hashlib
import heapq
import importlib
import inspect
import itertools
import json
import math
import numbers
import os
from pathlib import Path
import pdb
from pprint import pformat, pprint, PrettyPrinter
import random
import re
import shutil
import ssl
from struct import unpack
import sys
import textwrap
import threading
import torch
from torch import nn as torch_nn
from torch.nn import functional as torch_f
import traceback
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING, Union
# / BASIC IMPORTS

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from IPython.display import display, HTML
from jupyter_plotly_dash import JupyterDash
import numpy as np
import pandas as pd
import torch

class Visualization:
    """
    Keeps track of KPIs over the course of the Experiment and creates visualizations from them.
    """
    def __init__(self, visualization_params):
        super().__init__()
        self.app = None
        self.ranges_are_currently_active = False
        self.current_range = -1
        self.is_currently_at_start_of_range = False
        self.all_visualizable_objects = collections.OrderedDict()
        self.all_required_location_identifiers = {}
        self.dfs_per_range_and_location = None
        self.main_area_divs = None
        self.folder_for_graphs = None
        #
        # These are the only values that get serialized
        #
        # The iterations during which everything is tracked, so that details on those iterations
        # can be viewed in the Visualizer.
        self.iteration_ranges = []
        # The index of the action that was executed on a given iteration
        self.iteration_to_selected_action = {}
        # The stack_of_scenario_mechanism_instances at the beginning of each iteration
        self.iteration_to_stack_of_scenario_mechanism_instances = {}
        # Information for the ExperienceReplay
        self.object_name_to_iteration_to_summary = defaultdict(functools.partial(defaultdict, str))
        self.object_name_to_iteration_to_log_notes = defaultdict(functools.partial(defaultdict, list))
        # The version of each location at the end of each iteration
        self.iteration_to_location_to_latest_tensor_version = {}
        # KPIs always exist with exactly one value per iteration.
        # The mapping of this dictionary goes like this:
        # obj_name -> kpi_group_name -> kpi_name -> [value for each iteration]
        self.full_kpi_data = {}
        # This keeps track of all values extracted from locations, on each valid range.
        # This gets refined into a DataFrame later: self.dfs_per_range_and_location
        self.values_per_range_and_location = defaultdict(list)
        # Information about the ProcessVisualizableObjects
        self.processes_per_iteration = defaultdict(list)
        self.shared_process_indices_per_process_and_iteration = defaultdict(list)
        self.process_info_per_process_and_iteration = defaultdict(list)
        # Placement of the PUs on the circle
        self.pus_per_segment = defaultdict(list)

    def register_visualizable_object(self, obj):
        """
        Register a FixedVisualizableObject.
        """
        assert obj not in self.all_visualizable_objects, "Don't register an object twice."
        assert self.app is None, "This function should only be called BEFORE initialization."
        assert re.match("^[a-z_][a-zA-Z0-9_-]*$", obj.name), "Object names must not contain unusual characters, " \
                                                             "to ensure that string concatenation and split() " \
                                                             "functions don't break anything by accident:\n" \
                                                             f"{obj.name}"
        self.all_visualizable_objects[obj.name] = obj

    def get_coordinates_of_object(self, net: 'InteractionNetwork', identifier):
        """
        Helper function for other classes.
        Given either a location_identifier, or just the name of an object,
        returns the coordinates and the radius of that object.
        """
        if isinstance(identifier, str):
            obj = self.all_visualizable_objects[identifier]
            return obj.visualization_get_coordinates(net, self)
        else:
            obj = self.all_visualizable_objects[identifier[0]]
            return obj.visualization_get_contained_element_data_absolute(net, self, identifier)

    def initialize(self, net: 'InteractionNetwork', execution_mode):
        """
        Perform initialization after all FixedVisualizableObject have been registered.
        """
        assert net.params.visualization_enabled, "If visualization wasn't enabled, the visualizer should not exist."
        #
        # Initialize the iteration_ranges
        #
        # First, add any new iteration_ranges.
        # This starts at the current iteration and can count backwards from the expected last iteration of the run.
        starting_iteration = 1 + net.iteration
        end_iteration = starting_iteration + net.params.iterations_to_run
        new_iteration_ranges = net.params.new_iteration_ranges_to_record_for_visualization + \
            net.params.settings.extra_iteration_ranges_to_record_for_visualization
        new_ranges = []
        if execution_mode in ['run', 'test']:
            for s, e in new_iteration_ranges:
                if s >= 0:
                    s = starting_iteration + s
                else:
                    s = end_iteration + s + 1
                if e >= 0:
                    e = starting_iteration + e
                else:
                    e = end_iteration + e + 1
                s = max(min(s, end_iteration), starting_iteration)
                e = max(min(e, end_iteration), starting_iteration)
                new_ranges.append((s, e))
            # If the iteration ranges, are completely empty, add a dummy so that the Visualization is possible at all.
            if len(self.iteration_ranges) == 0:
                new_ranges.append((0, 1))
            # Sort the new ranges and merge neighbours that overlap into one
            new_ranges.sort()

            def recursive_merge(inter, start_index=0):
                for i in range(start_index, len(inter) - 1):
                    if inter[i][1] > inter[i + 1][0]:
                        new_start = inter[i][0]
                        new_end = inter[i + 1][1]
                        inter[i] = [new_start, new_end]
                        del inter[i + 1]
                        return recursive_merge(inter.copy(), start_index=i)
                return inter
            new_ranges = recursive_merge(new_ranges)
            self.iteration_ranges.extend(new_ranges)
            # Perform sanity checks to ensure the ranges are strictly increasing
            last_end = 0
            for s, e in self.iteration_ranges:
                assert s < e
                assert last_end <= s
                assert e <= end_iteration
                last_end = e
        #
        # Initialize the VisualizableObjects
        #
        for obj in self.all_visualizable_objects.values():
            obj.visualization_determine_size_and_location(net, self)
        #
        # Get all extraction_types for all location_identifiers that are required by any VisualizationObject.
        #
        for obj in self.all_visualizable_objects.values():
            for location_identifier, extraction_types in \
                    obj.visualization_get_location_data_extraction_types_needed_for_visualization(net, self):
                existing_extraction_types = self.all_required_location_identifiers.setdefault(location_identifier, [])
                for et in extraction_types:
                    if et not in existing_extraction_types:
                        existing_extraction_types.append(et)
        # Delete any entries where no extraction_types are required
        self.all_required_location_identifiers = {k: v for k, v in self.all_required_location_identifiers.items()
                                                  if len(v) > 0}
        # Initialize lists and dicts
        self.dfs_per_range_and_location = {}
        self.main_area_divs = []
        # Create and clear a folder for graphs if this is the first iteration
        self.folder_for_graphs = net.params.experiment_path / 'visualization_graphs'
        if net.iteration == -1:
            self.folder_for_graphs.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(self.folder_for_graphs)
            self.folder_for_graphs.mkdir()
        # Create the app that will later be used for visualization.
        self.app = JupyterDash(net.params.network_name, height=f'{net.params.visualization_params.total_height}px')

    def before_iteration(self, net: 'InteractionNetwork'):
        """
        Determine if it is the start of a new iteration_range.
        If it is the end of an iteration_range, stop gathering updates.
        Obtain the value for any required location_identifier of any VisualizationObject for the current iteration.
        Also track which SMIs are currently active.
        """
        # Did we just start or end an iteration_range?
        self.is_currently_at_start_of_range = False
        for i, (range_start, range_end) in enumerate(self.iteration_ranges):
            if net.iteration == range_start:
                self.is_currently_at_start_of_range = True
                self.ranges_are_currently_active = True
                self.current_range = i
            elif net.iteration == range_end:
                self.ranges_are_currently_active = False
                self.current_range = -1
        # If this Visualizer is currently active, update values for all locations,
        # and any other required metadata.
        if self.ranges_are_currently_active:
            for location_identifier in self.all_required_location_identifiers.keys():
                self._update_location_helper(net, location_identifier, was_an_update=False)
            self._track_scenario_mechanism_instances_per_iteration(net)

    def after_iteration(self, net: 'InteractionNetwork'):
        """
        Gather some more data about what the IN did this iteration.
        """
        if self.ranges_are_currently_active:
            self.iteration_to_selected_action[net.iteration] = \
                net.get_action_of_this_iteration()
            for obj in net.get_all_visualizable_objects():
                summary = obj.get_summary_for_visualization_on_this_iteration(net)
                if summary is not None:
                    if not isinstance(summary, str):
                        summary = utilities.StringFormatter()(summary)
                    self.object_name_to_iteration_to_summary[obj.name][net.iteration] = summary

    def _track_scenario_mechanism_instances_per_iteration(self, net: 'InteractionNetwork'):
        """
        Keep track of what SMIs are active.
        """
        a = net.stack_of_scenario_mechanism_instances
        # Make a copy of the SMI object, because the ScenarioMechanisms are allowed to modify it.
        # Note:
        # This could theoretically fail if someone stores a tensor.
        # That's allowed, but discouraged, because a copy is stored for every iteration
        # where self.ranges_are_currently_active.
        fields_to_copy = ['id', 'sm', 'start_iteration', 'start_age_iteration', 'age_iteration']
        a = [{k: v for k, v in b.items() if k in fields_to_copy} for b in a]
        a = json.loads(json.dumps(a))
        for b in a:
            b['is_a_simulation'] = net.scenario_mechanisms[b['sm']].__class__.Meta.is_a_simulation
        self.iteration_to_stack_of_scenario_mechanism_instances[net.iteration] = a

    def track_tensor_versions_per_iteration(self, net: 'InteractionNetwork'):
        """
        Keep track of what the latest version of each tensor was at the end of each iteration.
        """
        if not self.ranges_are_currently_active:
            return
        assert net.iteration not in self.iteration_to_location_to_latest_tensor_version
        location_to_latest_tensor_version = {}
        self.iteration_to_location_to_latest_tensor_version[net.iteration] = location_to_latest_tensor_version
        for obj_id, obj in self.all_visualizable_objects.items():
            for location in obj.get_all_owned_locations(net):
                version = net.tensor_database.get_version_of_location(net, location)
                location_to_latest_tensor_version[location] = version

    def gather_all_kpis(self, net: 'InteractionNetwork'):
        """
        Update all KPIs by querying all registered objects for their current KPI values.
        """
        # A helper function for initializing the KPI dictionary and verifying that it's used correctly.
        def _initialize_on_start_else_verify_equivalence(existing_dict, value, new_value_dict, init_func):
            if net.iteration == 0:
                assert value not in existing_dict
                existing_dict[value] = init_func()
            else:
                assert value in existing_dict
                if isinstance(new_value_dict, dict):
                    assert len(existing_dict[value]) == len(new_value_dict), \
                        f"{len(existing_dict[value])} {len(new_value_dict)}\n" \
                        f"{existing_dict[value]=}\n{new_value_dict=}"
            return existing_dict[value]
        # Get all KPI data from each registered object
        for obj_id, obj in self.all_visualizable_objects.items():
            new_obj_kpi_data = obj.visualization_get_current_kpis(net, self)
            assert isinstance(new_obj_kpi_data, dict)
            existing_kpi_group_data = _initialize_on_start_else_verify_equivalence(self.full_kpi_data,
                                                                                   obj_id,
                                                                                   new_obj_kpi_data,
                                                                                   dict)
            for kpi_group_name, new_kpi_group_data in new_obj_kpi_data.items():
                assert isinstance(new_kpi_group_data, dict)
                existing_kpi_data = _initialize_on_start_else_verify_equivalence(existing_kpi_group_data,
                                                                                 kpi_group_name,
                                                                                 new_kpi_group_data, dict)
                for kpi_name, new_kpi_value in new_kpi_group_data.items():
                    kpi_list = _initialize_on_start_else_verify_equivalence(existing_kpi_data,
                                                                            kpi_name,
                                                                            new_kpi_value,
                                                                            list)
                    assert len(kpi_list) == net.iteration, "Exactly one value for each KPI should be set per iteration"
                    kpi_list.append(new_kpi_value)

    def update_location(self, net: 'InteractionNetwork', location_identifier):
        """
        If this visualization is active, update the values for a given location_identifier
        """
        if not self.ranges_are_currently_active:
            return
        self._update_location_helper(net, location_identifier, was_an_update=True)

    def add_instance_of_process(self, net: 'InteractionNetwork', process_name, associated_object,
                                from_locations_and_versions, to_locations_and_versions, info):
        """
        If this visualization is active, record an entry of a process for a ProcessVisualizableObject.
        """
        if not self.ranges_are_currently_active:
            return
        process = (process_name, associated_object, from_locations_and_versions, to_locations_and_versions, info)
        self.shared_process_indices_per_process_and_iteration[(process_name, net.iteration)].append(
            len(self.processes_per_iteration[net.iteration]))
        self.processes_per_iteration[net.iteration].append(process)

    def add_process_info_for_details_view(self, net: 'InteractionNetwork', process_name, info):
        """
        If this visualization is active, record an entry of miscellaneous information for a ProcessVisualizableObject.
        """
        if not self.ranges_are_currently_active:
            return
        self.process_info_per_process_and_iteration[(process_name, net.iteration)].append(info)

    def _update_location_helper(self, net: 'InteractionNetwork', location_identifier, was_an_update=True):
        """
        Used by before_iteration() and update_location().
        Either extracts new KPIs from a location, or copies previous values of that location,
        depending on whether or not the location was updated.
        The resulting DataFrame has at least one row per iteration,
        and has exactly one row per iteration that is not marked as overwritten.
        """
        assert net.iteration not in self.iteration_to_location_to_latest_tensor_version, \
            f"The function track_tensor_versions_per_iteration() has already been called. " \
            f"There should be no further updates to tensors in this iteration.\n" \
            f"{net.iteration=}"
        val = net.tensor_database.get_value_at_location(net, location_identifier)
        version = net.tensor_database.get_version_of_location(net, location_identifier)
        smi = version.smi_id
        tensor_id = version.tensor_id
        age = net.get_age_iteration() - version[-2]
        assert age >= 0, \
            f"There should be no tensor accessible with an age_iteration that exceeds " \
            f"the age_iteration of the network.\n" \
            f"If there is, this is probably because SMIs are not implemented correctly and a tensor is not replaced " \
            f"after the SMI that created it is removed, and the age_iteration is reset.\n" \
            f"{net.get_age_iteration()}, {version}"
        key = (self.current_range, location_identifier)
        # During initialization, use a dummy tensor instead of None
        assert net.iteration != -1 or val is not None, f"Values can only be None during initialization:" \
                                                       f"{location_identifier}"
        if val is None or val.numel() == 0:
            val = torch.zeros((1,))
        # Get the previous value, if one exists, to determine if the value was updated.
        previous_values = self.values_per_range_and_location[key]
        if len(previous_values) == 0 or self.is_currently_at_start_of_range or \
                previous_values[-1]['_version'] != version:
            if location_identifier not in self.all_required_location_identifiers:
                log(f"{location_identifier} is not registered with visualizer", cond='warning')
                return
            # If the value did change, extract all required data
            extraction_types = self.all_required_location_identifiers[location_identifier]
            extracted_vals = {}
            for et in extraction_types:
                if et == 'tensor_kpis':
                    mean = val.mean().item()
                    std = val.std(False).item()
                    tmp = {
                        'max': val.max().item(),
                        'min': val.min().item(),
                        'mean': mean,
                        'mean+std': mean + std,
                        'mean-std': mean - std,
                        'l2/numel': torch.norm(val, 2) / val.numel(),
                    }
                elif et == 'full_tensor_data':
                    tmp = {
                        'tensor': val.detach().clone()
                    }
                else:
                    raise ValueError(et)
                extracted_vals.update(tmp)
            # If this was_an_update, get the previous value and mark it as overwritten
            if was_an_update:
                prev_val = previous_values[-1]
                prev_val['was_overwritten'] = True
        else:
            # If the value did not change since last time, copy it
            prev_val = previous_values[-1]
            extracted_vals = {k: v for k, v in prev_val.items()}
        # Find out if the SMI changed due to being imported with
        # tensor_database.renew_different_version_of_tensor_at_location().
        # (This results in a different version than the last tensor that was registered on the last iteration,
        # because tensor_database.renew_different_version_of_tensor_at_location()
        # should be called after the visualizer.)
        # Also mark its as such if a previous tensor of this iteration is marked as such.
        # This way, if a tensor is reset and then overwritten, it will show up in both colors.
        stealth_change = False
        if len(previous_values) > 0:
            prev_val = previous_values[-1]
            stealth_change = (version != prev_val['_version'] and prev_val['iteration'] != net.iteration) or \
                           (prev_val['stealth_change'] and prev_val['iteration'] == net.iteration)
        # Extract the required data and store it
        extracted_vals['tensor_id'] = tensor_id
        extracted_vals['smi'] = smi
        extracted_vals['age'] = age
        extracted_vals['_version'] = version
        extracted_vals['stealth_change'] = stealth_change
        extracted_vals['iteration'] = net.iteration
        extracted_vals['was_overwritten'] = False
        self.values_per_range_and_location[key].append(extracted_vals)

    def finalize(self, net: 'InteractionNetwork'):
        """
        Compile everything so that the app is ready to be displayed.
        """
        assert net.params.visualization_enabled, "If visualization wasn't enabled, the visualizer should not exist."
        app = self.app
        vp = net.params.visualization_params
        # Verify KPIs
        for obj_id, obj in self.all_visualizable_objects.items():
            for kpi_group_name, kpi_group_data in self.full_kpi_data[obj_id].items():
                for kpi_name, kpi_values in kpi_group_data.items():
                    assert len(kpi_values) == net.iteration + 1, "One KPI value should be set per iteration."
        # Compile self.values_per_range_and_location into self.dfs_per_range_and_location
        for range_and_location, extracted_vals_list in self.values_per_range_and_location.items():
            if len(extracted_vals_list) == 0:
                log(f"{range_and_location} has no items", cond='warning')
                continue
            extracted_vals_list = [{k: v for k, v in a.items() if not k.startswith('_')}
                                   for a in extracted_vals_list]
            df = pd.DataFrame(extracted_vals_list)
            self.dfs_per_range_and_location[range_and_location] = df
            # Sanity checks
            df_newest_only = df[~df['was_overwritten']]
            range_start, range_end = self.iteration_ranges[range_and_location[0]]
            iterations_in_range = range_end - range_start
            assert df_newest_only.shape[0] == iterations_in_range, \
                f"{df_newest_only.shape[0]}\n" \
                f"{iterations_in_range}\n" \
                f"{range_and_location}"
            assert df_newest_only['iteration'].nunique() == iterations_in_range, \
                f"{df_newest_only['iteration'].nunique()}\n" \
                f"{iterations_in_range}\n" \
                f"{df_newest_only['iteration']}"
        # For each VisualizableObject that has been registered...
        for e in self.all_visualizable_objects.values():
            # Define divs and callbacks
            e.visualization_add_divs(net, self)
        # Create extra control elements based on self.all_visualizable_objects
        all_extra_controls = [a
                              for b in self.all_visualizable_objects.values()
                              for a in b.create_extra_controls(net)]
        #
        # Define callbacks
        #

        # Set the names of the selected and highlighted objects

        @app.callback(Output('dummy_for_selecting_an_object', 'className'),
                      [Input(a, 'n_clicks_timestamp') for a
                       in self.all_visualizable_objects.keys()])
        def update_visibility(*lsts):
            num_objs = len(self.all_visualizable_objects)
            clicks_per_object = lsts[0:num_objs]
            # Identify the most recently clicked object
            max_index = 0
            max_val = 0
            for i, b in enumerate(clicks_per_object):
                if b is not None and b > max_val:
                    max_val = b
                    max_index = i
            name_of_selected_item = list(self.all_visualizable_objects.keys())[max_index]
            # For each object, apply status to it depending on whether or not it was selected
            # or is otherwise highlighted.
            selected_objects_status = {}
            for obj_div_id in self.all_visualizable_objects.keys():
                is_latest = obj_div_id == name_of_selected_item
                class_names = []
                if is_latest:
                    class_names.append('selected')
                    class_names.append('highlighted')
                selected_objects_status[obj_div_id] = class_names
            selected_objects_status = json.dumps(selected_objects_status)
            return selected_objects_status

        # Update the visuals of the main divs

        @app.callback([Output(obj_div_id, 'style') for obj_div_id
                       in self.all_visualizable_objects.keys()] +
                      [Output(obj_div_id, 'children') for obj_div_id
                       in self.all_visualizable_objects.keys()],
                      [Input('iteration_range_selector', 'value'),
                       Input('aggregation_range_selector', 'value'),
                       Input('current_time_slider', 'value'),
                       Input('dummy_for_selecting_an_object', 'className'),
                       Input('dummy_for_hovering_over_actions', 'className'),
                       Input(f'show_unused_pus', 'value'),
                       Input(f'show_unused_envs', 'value'),
                       Input(f'control_unit_selected_channels', 'value')] +
                      [Input(a.id, 'value') for a in all_extra_controls])
        def update_main_divs(irs_val, iteration_aggregation, selected_iteration, selected_objects_status,
                             hovered_action, show_unused_pus, show_unused_envs, control_unit_selected_channels,
                             *extra_controls_values):
            if irs_val is None or iteration_aggregation is None or selected_iteration is None:
                raise PreventUpdate()
            if selected_objects_status is None or selected_objects_status == '':
                selected_objects_status = defaultdict(list)
            else:
                selected_objects_status = json.loads(selected_objects_status)
            extra_control_values = {a.id: b for a, b in
                                    zip(all_extra_controls, extra_controls_values)}
            obj_styles_list = []
            obj_children_list = []
            # For each object...
            for obj in self.all_visualizable_objects.values():
                obj_children = []
                # Get the style of the main_div
                obj_style, extra_children = obj.visualization_create_main_div(net, self, selected_objects_status,
                                                                              irs_val, iteration_aggregation,
                                                                              selected_iteration, hovered_action,
                                                                              show_unused_pus, show_unused_envs,
                                                                              control_unit_selected_channels,
                                                                              extra_control_values)
                obj_styles_list.append(obj_style)
                obj_children.extend(extra_children)
                # Create the children of the main_div and set their style
                for location_identifier in obj.get_all_owned_locations(net):
                    name = f"{location_identifier[0]}__{location_identifier[1]}"
                    v = self.iteration_to_location_to_latest_tensor_version[selected_iteration][location_identifier]
                    title = f"{location_identifier[0]}, {location_identifier[1]} - SMI {v[0]}, iteration {v[1]}, " \
                            f"version in iteration {v[2]}, total version {v[3]}, age_iteration {v[4]}, ID {v[5]}"
                    child = html.Div(id=name, title=title)
                    obj_children.append(child)
                    obj.visualization_create_contained_element_div(net, self, location_identifier, child,
                                                                   selected_objects_status, iteration_aggregation,
                                                                   selected_iteration, irs_val,
                                                                   control_unit_selected_channels)
                obj_children_list.append(obj_children)
            return obj_styles_list + obj_children_list

        # Update the visuals of the details_div for the selected object

        @app.callback([Output('object_details_area', 'children'),
                       Output('current_time_slider_value_display', 'children')],
                      [Input('dummy_for_selecting_an_object', 'className'),
                       Input('iteration_range_selector', 'value'),
                       Input('aggregation_range_selector', 'value'),
                       Input('current_time_slider', 'value')])
        def update_details_area(selected_objects_status, irs_val, iteration_aggregation, selected_iteration):
            if selected_objects_status is None or selected_objects_status == '':
                selected_objects_status = defaultdict(list)
            else:
                selected_objects_status = json.loads(selected_objects_status)
            name_of_selected_object = None
            # Get the first selected object
            for k, v in selected_objects_status.items():
                if 'selected' in v:
                    name_of_selected_object = k
                    break
            if name_of_selected_object is None:
                return "Select an object to view details on it."
            obj = self.all_visualizable_objects[name_of_selected_object]
            res = obj.visualization_create_details_div(net, self,
                                                       irs_val, iteration_aggregation, selected_iteration)
            return [res, selected_iteration]

        # Make the action_activations of the CU selectable

        @app.callback(Output('dummy_for_hovering_over_actions', 'className'),
                      [Input('cu_action_activations_graph', 'hoverData')])
        def update_class_for_hovering(hover_data):
            if hover_data is None:
                return 'no_action_hovered,0'
            points = hover_data['points']
            if len(points) == 1:
                res = points[0]['text']
                # Remove trailing '__suffix' suffices
                if '__' in res:
                    res = res[:res.index('__')]
                return res
            return 'no_action_hovered,0'

        # Define the Layout of the App
        app.layout = html.Div(style={
            'backgroundColor': 'white',
        }, children=[
            html.Div(id='dummy_for_loading_when_only_input_element'),
            html.Div(id='dummy_for_selecting_an_object'),  # This dummy stores some data
            html.Div(id='dummy_for_hovering_over_actions'),  # This dummy stores some data
            html.H1(children=net.params.network_name),
            html.Div(style={
                'display': 'flex',
                'width': f'{vp.total_screen_width}px',
            }, children=[
                html.Div(id='main_area', style={
                    'width': f'{vp.main_area_width}px',
                    'height': f'{vp.main_area_height}px',
                    'backgroundColor': 'white',
                    'position': 'relative',
                    'border': '1px solid black',
                }, children=self.main_area_divs),
            ]),
            html.Div(style={
                'display': 'flex',
                'width': f'{vp.total_screen_width}px',
            }, children=all_extra_controls),
            html.Div(style={
                'display': 'flex',
                'width': f'{vp.total_screen_width}px',
            }, children=[
                html.Div(style={
                    'width': '200px',
                }, children=[
                    dcc.Dropdown(
                        id='iteration_range_selector',
                        options=[{'label': f"{range_start} to {range_end}", 'value': i}
                                 for i, (range_start, range_end) in enumerate(self.iteration_ranges)],
                        value=len(self.iteration_ranges)-1,
                        clearable=False
                    )
                ]),
                html.Div(style={
                    'width': '200px',
                }, children=[
                    dcc.Dropdown(
                        id='aggregation_range_selector',
                        options=[{'label': 'None' if i == 1 else str(i), 'value': i}
                                 for i in vp.aggregation_range_options],
                        value=1,
                        clearable=False
                    )
                ]),
                html.Div(style={
                    'width': f'{vp.total_screen_width - 200 - 200}px',
                }, children=[
                    dcc.Slider(
                        id='current_time_slider',
                        # The values here are set by a callback
                    ),
                    html.Div(id='current_time_slider_value_display', style={
                            'padding-left': '50%',
                     }, children='0',)
                ]),
            ]),
            html.Div(style={
                'display': 'flex',
                'width': f'{vp.total_screen_width}px',
            }, children=[
                dcc.Checklist(
                    id='show_unused_pus',
                    options=[
                        {'label': f'Show unused PUs', 'value': 'show'},
                    ],
                    value=[],
                ),
                dcc.Checklist(
                    id='show_unused_envs',
                    options=[
                        {'label': f'Show unused Environments', 'value': 'show'},
                    ],
                    value=[],
                ),
                dcc.Dropdown(
                    id='control_unit_selected_channels',
                    options=[{'label': v, 'value': i} for i, v in enumerate(net.get_all_reward_channels())],
                    value=[i for i, v in enumerate(net.get_all_reward_channels())],
                    multi=True,
                    clearable=False
                ),
                dcc.Input(
                    id="manual_input_for_current_time_slider",
                    type='number',
                    placeholder="0",
                ),
            ]),
            html.Div(style={
                'width': f'{vp.total_screen_width}px',
            }, children=[
                html.Div(id='object_details_area', style={
                    'height': f'{vp.object_details_area_height}px',
                    'overflow': 'auto',
                })
            ]),
        ])

        #
        # Make the current_time_slider depend on the iteration_range_selector
        #

        @app.callback(
            Output('current_time_slider', 'min'),
            [Input('iteration_range_selector', 'value')])
        def func(selected_range_index):
            return self.iteration_ranges[selected_range_index][0]

        @app.callback(
            Output('current_time_slider', 'max'),
            [Input('iteration_range_selector', 'value')])
        def func(selected_range_index):
            return self.iteration_ranges[selected_range_index][1] - 1

        @app.callback(
            Output('current_time_slider', 'marks'),
            [Input('current_time_slider', 'min'), Input('current_time_slider', 'max')])
        def func(min_val, max_val):
            if min_val is None or max_val is None:
                # This will be overwritten soon when this function is called again.
                # In the meantime, this avoids printing an error message to the console.
                return {}
            step_size = int(math.ceil((max_val - min_val + 1) / 10))
            res = {i: str(i) for i in range(min_val, max_val + 1) if
                   i in [min_val, max_val] or (i % step_size == 0 and max_val - i + 1 >= step_size)}
            return res

        @app.callback(
            Output('current_time_slider', 'value'),
            [Input('manual_input_for_current_time_slider', 'n_submit'),
             Input('current_time_slider', 'min'),
             Input('current_time_slider', 'max')],
            [State('manual_input_for_current_time_slider', 'value')])
        def func(manual_value_was_submitted, min_val, max_val, manual_value):
            if isinstance(manual_value, int) and min_val <= manual_value <= max_val:
                return manual_value
            return max_val

    def display(self):
        """
        Display the finished app.
        """
        display(self.app)

    def get_save_state(self, net: 'InteractionNetwork'):
        iteration_to_location_to_latest_tensor_version = {k: {a: list(b) for a, b in v.items()} for k, v in
                                                          self.iteration_to_location_to_latest_tensor_version.items()}
        shared_process_indices = [[list(k), v] for k, v
                                  in self.shared_process_indices_per_process_and_iteration.items()]
        process_info_per_process_and_iteration = [[list(k), v] for k, v
                                                  in self.process_info_per_process_and_iteration.items()]
        data = {
            'iteration_ranges': self.iteration_ranges,
            'iteration_to_selected_action': self.iteration_to_selected_action,
            'iteration_to_stack_of_scenario_mechanism_instances': self.iteration_to_stack_of_scenario_mechanism_instances,
            'object_name_to_iteration_to_summary': self.object_name_to_iteration_to_summary,
            'object_name_to_iteration_to_log_notes': self.object_name_to_iteration_to_log_notes,
            'iteration_to_location_to_latest_tensor_version': iteration_to_location_to_latest_tensor_version,
            'full_kpi_data': self.full_kpi_data,
            'values_per_range_and_location': self.values_per_range_and_location,
            'processes_per_iteration': self.processes_per_iteration,
            'shared_process_indices_per_process_and_iteration': shared_process_indices,
            'process_info_per_process_and_iteration': process_info_per_process_and_iteration,
        }
        return data


def load_visualizer(data, net: 'InteractionNetwork'):
    res = Visualization(net.params.visualization_params)
    res.iteration_ranges = data['iteration_ranges']
    res.iteration_to_selected_action = data['iteration_to_selected_action']
    res.iteration_to_stack_of_scenario_mechanism_instances = data['iteration_to_stack_of_scenario_mechanism_instances']
    res.object_name_to_iteration_to_summary = data['object_name_to_iteration_to_summary']
    res.object_name_to_iteration_to_log_notes = data['object_name_to_iteration_to_log_notes']
    for k, v in data['iteration_to_location_to_latest_tensor_version'].items():
        res.iteration_to_location_to_latest_tensor_version[k] = {a: tuple(b) for a, b in v.items()}
    for k, v in data['full_kpi_data'].items():
        res.full_kpi_data[k] = v
    for k, v in data['values_per_range_and_location'].items():
        res.values_per_range_and_location[k] = v
    for k, v in data['processes_per_iteration'].items():
        res.processes_per_iteration[k] = v
    for kv in data['shared_process_indices_per_process_and_iteration']:
        k = tuple(kv[0])
        v = kv[1]
        res.shared_process_indices_per_process_and_iteration[k] = v
    for kv in data['process_info_per_process_and_iteration']:
        k = tuple(kv[0])
        v = kv[1]
        res.process_info_per_process_and_iteration[k] = v
    return res