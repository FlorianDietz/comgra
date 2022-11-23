# BASIC IMPORTS
import dataclasses
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
import pickle
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
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_svg
import numpy as np
import plotly.express as px
import pandas as pd
import torch

from comgra.objects import StatusAndGraph, ModuleRepresentation, ParameterRepresentation, TensorRepresentation, TensorRecordings, TensorMetadata
from comgra import utilities

utilities.PRINT_EACH_TIME = True

DISPLAY_CONNECTIONS_GRAPHICALLY = False
DISPLAY_NAMES_ON_NODES_GRAPHICALLY = False

LOCK_FOR_RECORDINGS = threading.Lock()


@dataclass
class VisualizationParameters:
    total_display_width = 1800
    total_display_height = 400
    padding_left = 50
    padding_right = 50
    padding_top = 50
    padding_bottom = 50
    ratio_of_space_between_nodes_to_node_size = 2.0
    node_type_to_color = {
        'input': '#60c156',
        'target': '#068989',
        'output': '#00ffff',
        'loss': '#c74fb7',
        'intermediate': '#a29fc9',
        'parameter': '#cfc100',
    }
    highlighting_colors = {
        'selected': '#ff1700',
        'highlighted': '#9a0000',
    }


vp = VisualizationParameters()


class Visualization:
    """
    Keeps track of KPIs over the course of the Experiment and creates visualizations from them.
    """
    def __init__(self, path):
        super().__init__()
        self.path: Path = path
        assert path.exists(), path
        assets_path = Path(__file__).absolute().parent.parent / 'assets'
        assert assets_path.exists(), "If this fails, files have been moved."
        self.app = dash.Dash(__name__, assets_folder=str(assets_path))
        self.configuration_type_to_status_and_graph: Dict[str, StatusAndGraph] = {}
        self.configuration_type_to_node_to_corners: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
        self.configuration_type_to_grid_of_nodes: Dict[str, List[List[str]]] = {}
        self._sanity_check_cache_to_avoid_duplicates = {}
        self.cache_for_tensor_recordings = {}
        self.attribute_selection_fallback_values = collections.defaultdict(list)

    def _node_name_to_dash_id(self, configuration_type, node_name):
        conversion = node_name.replace('.', '__')
        conversion = f"confnode__{configuration_type}__{conversion}"
        assert (conversion not in self._sanity_check_cache_to_avoid_duplicates or
               self._sanity_check_cache_to_avoid_duplicates[conversion] == node_name), \
            f"Two nodes have the same representation: " \
            f"{node_name}, {self._sanity_check_cache_to_avoid_duplicates[conversion]}, {conversion}"
        self._sanity_check_cache_to_avoid_duplicates[conversion] = node_name
        return conversion

    def _nodes_to_connection_dash_id(self, configuration_type, source, target):
        return f"connection__{self._node_name_to_dash_id(configuration_type, source)}__" \
               f"{self._node_name_to_dash_id(configuration_type, target)}"

    def run_server(self, port):
        #
        # Load data that can only be loaded once, because Divs depend on it.
        #
        for config_folder in (self.path / 'configs').iterdir():
            configuration_type = config_folder.name
            with open(config_folder / 'status_and_graph.pkl', 'rb') as f:
                status_and_graph: StatusAndGraph = pickle.load(f)
            self.configuration_type_to_status_and_graph[configuration_type] = status_and_graph
        #
        # Visualize
        #
        self.create_visualization()
        self.app.run_server(debug=True, port=port)

    @utilities.runtime_analysis_decorator
    def get_recordings_with_caching(self, trials_value) -> TensorRecordings:
        with LOCK_FOR_RECORDINGS:
            key = (trials_value,)
            recordings, num_training_steps = self.cache_for_tensor_recordings.get(key, (None, 0))
            recordings_path = self.path / 'trials' / trials_value / 'recordings'
            recording_files = sorted(list(recordings_path.iterdir()))
            if len(recording_files) > num_training_steps:
                for subfiles in recording_files:
                    subfiles = sorted(list(subfiles.iterdir()))
                    for recording_file in subfiles:
                        with open(recording_file, 'r') as f:
                            new_recordings_json = json.load(f)
                            new_recordings: TensorRecordings = TensorRecordings(**new_recordings_json)
                            new_recordings.recordings = utilities.PseudoDb([]).deserialize(new_recordings.recordings)
                            new_recordings.training_step_to_iteration_to_configuration_type = {
                                int(k1): {
                                    int(k2): v2 for k2, v2 in v1.items()
                                }
                                for k1, v1 in new_recordings.training_step_to_iteration_to_configuration_type.items()
                            }
                            new_recordings.node_to_role_to_tensor_metadata = {
                                k1: {
                                    k2: TensorMetadata(**v2) for k2, v2 in v1.items()
                                }
                                for k1, v1 in new_recordings.node_to_role_to_tensor_metadata.items()
                            }
                            if recordings is None:
                                recordings = new_recordings
                            else:
                                recordings.update_with_more_recordings(new_recordings)
                num_training_steps = len(recording_files)
                self.cache_for_tensor_recordings[key] = (recordings, num_training_steps)
            return recordings

    @utilities.runtime_analysis_decorator
    def create_nodes_and_arrows(
            self, configuration_type: str, sag: StatusAndGraph
    ) -> List:
        assert configuration_type not in self.configuration_type_to_node_to_corners, configuration_type
        node_to_corners = self.configuration_type_to_node_to_corners.setdefault(configuration_type, {})
        grid_of_nodes = self.configuration_type_to_grid_of_nodes.setdefault(configuration_type, [])
        highest_number_of_nodes = max(len(a) for a in sag.dag_format)
        height_per_box = int((vp.total_display_height - vp.padding_top - vp.padding_bottom) / (highest_number_of_nodes + (highest_number_of_nodes - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        width_per_box = int((vp.total_display_width - vp.padding_left - vp.padding_right) / (len(sag.dag_format) + (len(sag.dag_format) - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        elements_of_the_graph = []
        for i, nodes in enumerate(sag.dag_format):
            list_of_nodes_for_grid = []
            grid_of_nodes.append(list_of_nodes_for_grid)
            for j, node in enumerate(nodes):
                list_of_nodes_for_grid.append(node)
                left = int(vp.padding_left + i * (1 + vp.ratio_of_space_between_nodes_to_node_size) * width_per_box)
                top = int(vp.padding_top + j * (1 + vp.ratio_of_space_between_nodes_to_node_size) * height_per_box)
                right = int(left + width_per_box)
                bottom = int(top + height_per_box)
                node_to_corners[node] = (left, top, right, bottom)
                node_type = sag.name_to_node[node].type_of_tensor
                elements_of_the_graph.append(html.Div(id=self._node_name_to_dash_id(configuration_type, node), style={
                    'width': f'{width_per_box}px',
                    'height': f'{height_per_box}px',
                    'backgroundColor': 'white',
                    'position': 'absolute',
                    'left': f'{left}px',
                    'top': f'{top}px',
                    'background': vp.node_type_to_color[node_type]
                }, title=node[len("node__"):], children=(
                    [f'{node[len("node__"):]}'] if DISPLAY_NAMES_ON_NODES_GRAPHICALLY else [])
                ))
        connection_names_to_source_and_target = {}
        node_to_incoming_and_outgoing_lines = {n: ([], []) for n in sag.nodes}
        svg_connection_lines = []
        if DISPLAY_CONNECTIONS_GRAPHICALLY:
            for connection in sag.connections:
                source, target = tuple(connection)
                source_left, source_top, _, _ = node_to_corners[source]
                target_left, target_top, _, _ = node_to_corners[target]
                source_x = int(source_left + width_per_box)
                source_y = int(source_top + 0.5 * height_per_box)
                target_x = int(target_left)
                target_y = int(target_top + 0.5 * height_per_box)
                connection_name = self._nodes_to_connection_dash_id(configuration_type, source, target)
                svg_connection_lines.append(dash_svg.Line(
                    id=connection_name,
                    x1=str(source_x), x2=str(target_x), y1=str(source_y), y2=str(target_y),
                    stroke=vp.node_type_to_color[sag.name_to_node[source].type_of_tensor],
                    strokeWidth=1,
                ))
                connection_names_to_source_and_target[connection_name] = (source, target)
                node_to_incoming_and_outgoing_lines[source][1].append(connection_name)
                node_to_incoming_and_outgoing_lines[target][0].append(connection_name)
        elements_of_the_graph.append(dash_svg.Svg(svg_connection_lines, viewBox=f'0 0 {vp.total_display_width} {vp.total_display_height}'))
        return elements_of_the_graph

    @utilities.runtime_analysis_decorator
    def create_visualization(self):
        app = self.app
        # Define the Layout of the App
        app.layout = html.Div(style={
            'backgroundColor': 'white',
        }, children=[
            html.Div(id='dummy-for-selecting-a-node'),  # This dummy stores some data
            html.Div(id='graph-container', style={
                'position': 'relative',
                'width': f'{vp.total_display_width}px',
                'height': f'{vp.total_display_height}px',
                'border': '1px solid black',
            }, children=[
                html.Div(id=f'graph_container__{configuration_type}', style={
                        'position': 'relative',
                        'width': f'{vp.total_display_width}px',
                        'height': f'{vp.total_display_height}px',
                    }, children=self.create_nodes_and_arrows(configuration_type, sag)
                )
                for configuration_type, sag in self.configuration_type_to_status_and_graph.items()
            ] + [
                html.Div(id='graph-overlay-for-selections', style={
                        'position': 'absolute',
                        'top': '0',
                        'left': '0',
                        'width': f'{vp.total_display_width}px',
                        'height': f'{vp.total_display_height}px',
                        'pointer-events': 'none',
                    }
                )
            ]),
            dcc.Tooltip(id="graph-tooltip"),
            html.Div(id='controls-container', children=[
                html.Div(id='controls-buttons-container', children=[
                    html.Button('Refresh', id='refresh-button', n_clicks=0),
                    html.Button('Left', id='navigate-left-button', n_clicks=0),
                    html.Button('Right', id='navigate-right-button', n_clicks=0),
                    html.Button('Up', id='navigate-up-button', n_clicks=0),
                    html.Button('Down', id='navigate-down-button', n_clicks=0),
                ]),
                dcc.Dropdown(id='trials-dropdown', options=[], value=None),
                dcc.Slider(id='training-step-slider', min=0, max=100, step=None, value=None),
                dcc.RadioItems(id='type-of-recording-radio-buttons', options=[], value=None),
                dcc.Dropdown(id='batch-index-dropdown', options=[], value=None),
                dcc.Slider(id='iteration-slider', min=0, max=0, step=1, value=0),
                dcc.Dropdown(id='role-of-tensor-in-node-dropdown', options=[], value=None),
            ]),
            html.Div(id='selected-item-details-container', children=[
            ]),
        ])

        @app.callback([Output('trials-dropdown', 'options'),
                       Output('trials-dropdown', 'value')],
                      [Input('refresh-button', 'n_clicks')])
        @utilities.runtime_analysis_decorator
        def refresh_all(n_clicks):
            # Reset caches
            self.cache_for_tensor_recordings = {}
            # Load the list of trials
            trials_folder = self.path / 'trials'
            subfolders = [a for a in trials_folder.iterdir() if a.is_dir()]
            options = [{'label': a.name, 'value': a.name} for a in subfolders]
            options.sort(key=lambda a: a['label'])
            return options, options[0]['value']
        refresh_all(0)

        @app.callback(([Output('training-step-slider', 'min'),
                       Output('training-step-slider', 'max'),
                       Output('training-step-slider', 'marks'),
                       Output('training-step-slider', 'value'),
                       Output('type-of-recording-radio-buttons', 'options'),
                       Output('type-of-recording-radio-buttons', 'value'),
                       Output('batch-index-dropdown', 'options'),
                       Output('batch-index-dropdown', 'value'),
                       Output('iteration-slider', 'min'),
                       Output('iteration-slider', 'max'),
                       Output('iteration-slider', 'marks'),
                       Output('iteration-slider', 'value'),
                       Output('role-of-tensor-in-node-dropdown', 'options'),
                       Output('role-of-tensor-in-node-dropdown', 'value')] +
                      [Output(f'graph_container__{configuration_type}', 'className')
                       for configuration_type in self.configuration_type_to_status_and_graph.keys()]),
                      [Input('dummy-for-selecting-a-node', 'className'),
                       Input('trials-dropdown', 'value'),
                       Input('training-step-slider', 'value'),
                       Input('type-of-recording-radio-buttons', 'value'),
                       Input('batch-index-dropdown', 'value'),
                       Input('iteration-slider', 'value'),
                       Input('role-of-tensor-in-node-dropdown', 'value')])
        @utilities.runtime_analysis_decorator
        def update_dropboxes(
                name_of_selected_node,
                trials_value, training_step_value, type_of_recording_value, batch_index_value,
                iteration_value, role_of_tensor_in_node_value
        ):
            recordings = self.get_recordings_with_caching(trials_value)

            def create_slider_data_from_list(value, options_list):
                assert value in options_list, (value, options_list)
                options_list = sorted(options_list)
                marks = {a: str(a) for a in options_list}
                min_, max_ = (options_list[0], options_list[-1]) if options_list else (0, 100)
                return min_, max_, marks, value

            def create_options_and_value_from_list(value, options_list, label_maker=lambda a: str(a)):
                assert value in options_list, (value, options_list)
                options = [{'label': label_maker(a), 'value': a} for a in options_list]
                return options, value

            def query_database_using_current_values(attributes_to_ignore, current_params_dict_for_querying_database):
                filters = {}
                for name, val in current_params_dict_for_querying_database.items():
                    if val is not None and name not in attributes_to_ignore:
                        filters[name] = val
                list_of_matches, possible_attribute_values = db.get_matches(filters)
                return list_of_matches, possible_attribute_values
            #
            # Selection of valid values for control elements.
            #
            # Training steps come from file names,
            # while all other attributes are held by the contents of the files that training steps are named after.
            recordings_path = self.path / 'trials' / trials_value / 'recordings'
            recording_files = list(recordings_path.iterdir())
            training_steps = sorted([int(a.stem) for a in recording_files])
            assert len(training_steps) > 0
            training_step_min, training_step_max, training_step_marks, training_step_value = create_slider_data_from_list(
                training_step_value if training_step_value is not None else training_steps[0], training_steps)
            #
            # Query the database to determine the best-fitting record to set the current value.
            #
            db: utilities.PseudoDb = recordings.recordings
            current_params_dict_for_querying_database = {
                'training_step': training_step_value,
                'type_of_tensor_recording': type_of_recording_value,
                'batch_aggregation': batch_index_value,
                'iteration': iteration_value,
                'node_name': name_of_selected_node,
                'role_within_node': role_of_tensor_in_node_value,
                'item': None,
                'metadata': None,
            }
            list_of_matches = []
            possible_attribute_values = {}
            attributes_to_ignore_in_order = ['role_within_node', 'batch_aggregation', 'type_of_tensor_recording', 'training_step', 'iteration']
            for i in range(len(attributes_to_ignore_in_order) + 1):
                # If it is not possible to find a match, set the filters to None one by one until a match is found.
                # This can happen e.g. if you select a different node
                # and there is no role_within_node for which that is valid
                list_of_matches, possible_attribute_values = query_database_using_current_values(
                    attributes_to_ignore_in_order[:i], current_params_dict_for_querying_database
                )
                if list_of_matches:
                    break
            assert list_of_matches
            selected_record_values = tuple(list_of_matches[0][0])
            for attr, val in zip(db.attributes, selected_record_values):
                assert attr in current_params_dict_for_querying_database, attr
                current_params_dict_for_querying_database[attr] = val
            # Use fallback values if possible.
            # This is useful if you e.g. switch between nodes in such a way that some selections are temporarily invalid,
            # but you would like to return to your previous selection when you select a previously selected node again.
            # Logic: Switch to the last possible value in the fallback list, unless that is the very last element.
            # Then update the fallback list by removing all alternative values, and add the new selected value to the end of it.
            # TODO batch 0. right. left. batch mean. --> [0, has_no__batch_dimension, 0]
            print(123)
            attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid = ['iteration', 'batch_aggregation', 'role_within_node']
            for attr in attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid:
                print(123, attr)
                print(current_params_dict_for_querying_database)
                fallback_list = self.attribute_selection_fallback_values[attr]
                val = current_params_dict_for_querying_database[attr]
                value_to_switch_to = next((a for a in fallback_list[::-1] if a in possible_attribute_values[attr]), None)
                print(fallback_list)
                print(val, value_to_switch_to)
                if value_to_switch_to is not None and val != value_to_switch_to and value_to_switch_to != fallback_list[-1]:
                    print('switch', attr, val, value_to_switch_to)
                    current_params_dict_for_querying_database[attr] = value_to_switch_to
                    list_of_matches, possible_attribute_values = query_database_using_current_values(
                        [], current_params_dict_for_querying_database
                    )
                    assert list_of_matches
                    selected_record_values = tuple(list_of_matches[0][0])
                    for attr, val in zip(db.attributes, selected_record_values):
                        assert attr in current_params_dict_for_querying_database, attr
                        current_params_dict_for_querying_database[attr] = val
                for a in possible_attribute_values[attr]:
                    while a in fallback_list:
                        fallback_list.remove(a)
            for attr in attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid:
                fallback_list = self.attribute_selection_fallback_values[attr]
                val = current_params_dict_for_querying_database[attr]
                fallback_list.append(val)
            print('attribute_selection_fallback_values', self.attribute_selection_fallback_values)
            # Get the values of the selected record
            training_step_value, type_of_recording_value, batch_index_value, iteration_value, name_of_selected_node, role_of_tensor_in_node_value, item, metadata = selected_record_values
            #
            # Query again, using that record as the filter,
            # to determine which alternative values are legal for each attribute.
            #
            print(999)
            print(selected_record_values)
            print(current_params_dict_for_querying_database)
            _, possible_attribute_values = query_database_using_current_values(
                [], current_params_dict_for_querying_database
            )
            print(possible_attribute_values)
            # Get the values to return
            type_of_recording_options, type_of_recording_value = create_options_and_value_from_list(
                type_of_recording_value, possible_attribute_values['type_of_tensor_recording'],
            )
            type_of_recording_options.sort(key=lambda a: a['value'])
            batch_index_options, batch_index_value = create_options_and_value_from_list(
                batch_index_value, possible_attribute_values['batch_aggregation'],
                label_maker=lambda a: "Mean over the batch" if a == 'batch_mean' else ("Has no batch dimension" if a == 'has_no_batch_dimension' else f"batch index {a}")
            )
            batch_index_options.sort(key=lambda a: -1 if a['value'] == 'batch_mean' else (-2 if a['value'] == 'has_no_batch_dimension' else a['value']))
            iteration_min, iteration_max, iteration_marks, iteration_value = create_slider_data_from_list(
                iteration_value, possible_attribute_values['iteration'],
            )
            role_of_tensor_in_node_options, role_of_tensor_in_node_value = create_options_and_value_from_list(
                role_of_tensor_in_node_value, possible_attribute_values['role_within_node'],
            )
            role_of_tensor_in_node_options.sort(key=lambda a: a['value'])
            #
            # Hide or show different graphs
            #
            configuration_type = recordings.training_step_to_iteration_to_configuration_type[training_step_value][iteration_value]
            graph_container_visibilities = [
                'active' if conf_type == configuration_type else 'inactive'
                for conf_type in self.configuration_type_to_status_and_graph.keys()
            ]
            res = [
                      training_step_min, training_step_max, training_step_marks, training_step_value,
                      type_of_recording_options, type_of_recording_value, batch_index_options, batch_index_value,
                      iteration_min, iteration_max, iteration_marks, iteration_value,
                      role_of_tensor_in_node_options, role_of_tensor_in_node_value,
                  ] + graph_container_visibilities
            return res

        @app.callback(Output('dummy-for-selecting-a-node', 'className'),
                      [Input('trials-dropdown', 'value'),
                       Input('training-step-slider', 'value'),
                       Input('iteration-slider', 'value'),
                       Input('dummy-for-selecting-a-node', 'className'),
                       Input('navigate-left-button', 'n_clicks_timestamp'),
                       Input('navigate-right-button', 'n_clicks_timestamp'),
                       Input('navigate-up-button', 'n_clicks_timestamp'),
                       Input('navigate-down-button', 'n_clicks_timestamp')] +
                      [Input(self._node_name_to_dash_id(configuration_type, node), 'n_clicks_timestamp')
                       for configuration_type, sag in self.configuration_type_to_status_and_graph.items()
                       for node in sag.name_to_node.keys()])
        @utilities.runtime_analysis_decorator
        def update_visibility(*lsts):
            trials_value = lsts[0]
            training_step_value = lsts[1]
            iteration_value = lsts[2]
            previous_name_of_selected_node = lsts[3]
            navigation_button_clicks = lsts[4:8]
            clicks_per_node = lsts[8:]
            recordings = self.get_recordings_with_caching(trials_value)
            names = [
                node
                for _, sag in self.configuration_type_to_status_and_graph.items()
                for node in sag.name_to_node.keys()
            ]
            selected_configuration_type = recordings.training_step_to_iteration_to_configuration_type[training_step_value][iteration_value]
            names_that_exist_in_the_selected_configuration = {
                node for node in
                self.configuration_type_to_status_and_graph[selected_configuration_type].name_to_node.keys()
            }
            assert len(clicks_per_node) == len(names)
            # Identify the most recently clicked navigation button
            max_index_navigation = -1
            max_val_navigation = 0
            for i, clicks in enumerate(navigation_button_clicks):
                if clicks is not None and clicks > max_val_navigation:
                    max_val_navigation = clicks
                    max_index_navigation = i
            # Identify the most recently clicked nodes
            max_index_nodes = [0 if a is None else 1 for a in names].index(1)
            max_val_nodes = 0
            for i, (clicks, name) in enumerate(zip(clicks_per_node, names)):
                if name in names_that_exist_in_the_selected_configuration and clicks is not None and clicks > max_val_nodes:
                    max_val_nodes = clicks
                    max_index_nodes = i
            # Select a node based either on navigation buttons or based on clicking on that node,
            # whichever happened more recently.
            if max_val_nodes >= max_val_navigation:
                name_of_selected_node = names[max_index_nodes]
            else:
                assert previous_name_of_selected_node is not None
                grid_of_nodes = self.configuration_type_to_grid_of_nodes[selected_configuration_type]
                x, y = None, None
                for x, column in enumerate(grid_of_nodes):
                    for y, nn in enumerate(column):
                        if nn == previous_name_of_selected_node:
                            break
                    else:
                        continue
                    break
                # left / right
                if max_index_navigation == 0:
                    x -= 1
                elif max_index_navigation == 1:
                    x += 1
                if x < 0:
                    x = len(grid_of_nodes) - 1
                if x >= len(grid_of_nodes):
                    x = 0
                column = grid_of_nodes[x]
                if y >= len(column):
                    y = len(column) - 1
                # up / down
                if max_index_navigation == 2:
                    y -= 1
                elif max_index_navigation == 3:
                    y += 1
                if y < 0:
                    y = len(column) - 1
                if y >= len(column):
                    y = 0
                name_of_selected_node = column[y]
            assert name_of_selected_node is not None
            return name_of_selected_node

        @app.callback(
            [Output('selected-item-details-container', 'children'),
             Output('graph-overlay-for-selections', 'children')],
            [Input('dummy-for-selecting-a-node', 'className'),
             Input('trials-dropdown', 'value'),
             Input('training-step-slider', 'value'),
             Input('type-of-recording-radio-buttons', 'value'),
             Input('batch-index-dropdown', 'value'),
             Input('iteration-slider', 'value'),
             Input('role-of-tensor-in-node-dropdown', 'value')]
        )
        @utilities.runtime_analysis_decorator
        def select_node(
                node_name, trials_value, training_step_value, type_of_recording_value,
                batch_index_value, iteration_value, role_of_tensor_in_node_value,
        ):
            recordings = self.get_recordings_with_caching(trials_value)
            configuration_type = recordings.training_step_to_iteration_to_configuration_type[training_step_value][iteration_value]
            sag = self.configuration_type_to_status_and_graph[configuration_type]
            #
            # Select the node and visually highlight it.
            #
            # NOTE
            # This works in a hacky way, by constructing an SVG in graph-overlay-for-selections
            # and drawing on that overlay to highlight the nodes.
            # It would be cleaner to do this by adding a CSS class to the Nodes DIVs.
            # I already tried this. Unfortunately, Dash appears to be unable to handle this efficiently.
            # Even though the className of Nodes was not used as a dependency for anything,
            # the Javascript part of the code took a very long time to run,
            # which was proportional to the number of nodes.
            # This made this approach infeasible in practice.
            #
            node = sag.name_to_node[node_name]
            connected_node_names = {a[0] for a in sag.connections if a[1] == node_name} | {a[1] for a in sag.connections if a[0] == node_name}
            graph_overlay_elements = []
            for node_name_, (left, top, right, bottom) in self.configuration_type_to_node_to_corners[configuration_type].items():
                if node_name_ == node_name:
                    color = vp.highlighting_colors['selected']
                elif node_name_ in connected_node_names:
                    color = vp.highlighting_colors['highlighted']
                else:
                    continue
                corners = [
                    (left - 1, right + 1, top, top),
                    (right, right, top - 1, bottom + 1),
                    (left - 1, right + 1, bottom, bottom),
                    (left, left, top - 1, bottom + 1),
                ]
                for x1, x2, y1, y2 in corners:
                    graph_overlay_elements.append(dash_svg.Line(
                        x1=str(x1), x2=str(x2), y1=str(y1), y2=str(y2),
                        stroke=color,
                        strokeWidth=3,
                    ))
            graph_overlay_for_selections_children = [
                dash_svg.Svg(
                    viewBox=f'0 0 {vp.total_display_width} {vp.total_display_height}',
                    children=graph_overlay_elements
                )
            ]
            #
            # Display values based on the selected node
            #
            db: utilities.PseudoDb = recordings.recordings
            filters = {}
            for name, val in [
                ('training_step', training_step_value),
                ('type_of_tensor_recording', type_of_recording_value),
                ('batch_aggregation', batch_index_value),
                ('iteration', iteration_value),
                ('node_name', node_name),
                ('role_within_node', role_of_tensor_in_node_value),
                ('item', None),
                ('metadata', None),
            ]:
                if val is not None:
                    filters[name] = val
                assert name in ['item', 'metadata'] or val is not None, (name,)
            list_of_matches, possible_attribute_values = db.get_matches(filters)
            print(888, filters)
            print(possible_attribute_values)
            assert len(list_of_matches) > 0
            rows = []
            index_of_item = db.attributes.index('item')
            index_of_metadata = db.attributes.index('metadata')
            for key, val in list_of_matches:
                row = [
                    html.Td(key[index_of_item]),
                    html.Td(key[index_of_metadata]),
                    html.Td(val),
                ]
                rows.append(html.Tr(row))
            desc_text = node.type_of_tensor
            tensor_shape = recordings.node_to_role_to_tensor_metadata[node.full_unique_name][role_of_tensor_in_node_value].shape
            children = [
                html.Header(f"{node.full_unique_name} - {role_of_tensor_in_node_value}"),
                html.P(desc_text),
                html.P(f"Shape: [{', '.join([str(a) for a in tensor_shape])}]"),
                html.Table([html.Tr([html.Th(col) for col in ['KPI', 'metadata', 'value']])] + rows)
            ]
            return children, graph_overlay_for_selections_children
