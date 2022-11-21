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

from comgra.objects import StatusAndGraph, ModuleRepresentation, ParameterRepresentation, TensorRepresentation, TensorRecordings
from comgra import utilities

utilities.PRINT_EACH_TIME = False

DISPLAY_CONNECTIONS_GRAPHICALLY = False
DISPLAY_NAMES_ON_NODES_GRAPHICALLY = False

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
        self._sanity_check_cache_to_avoid_duplicates = {}
        self.cache_for_tensor_recordings = {}

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
        key = (trials_value,)
        recordings, num_training_steps = self.cache_for_tensor_recordings.get(key, (None, 0))
        recordings_path = self.path / 'trials' / trials_value / 'recordings'
        recording_files = list(recordings_path.iterdir())
        if len(recording_files) > num_training_steps:
            for recording_file in recording_files:
                with open(recording_file, 'rb') as f:
                    new_recordings: TensorRecordings = pickle.load(f)
                    if recordings is None:
                        recordings = new_recordings
                    else:
                        recordings.update_with_more_recordings(new_recordings)
            num_training_steps = len(recording_files)
            self.cache_for_tensor_recordings[key] = (recordings, num_training_steps)
        assert num_training_steps == len(recordings.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records)
        return recordings

    @utilities.runtime_analysis_decorator
    def create_nodes_and_arrows(
            self, configuration_type: str, sag: StatusAndGraph
    ) -> List:
        assert configuration_type not in self.configuration_type_to_node_to_corners, configuration_type
        node_to_corners = self.configuration_type_to_node_to_corners.setdefault(configuration_type, {})
        highest_number_of_nodes = max(len(a) for a in sag.dag_format)
        height_per_box = int((vp.total_display_height - vp.padding_top - vp.padding_bottom) / (highest_number_of_nodes + (highest_number_of_nodes - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        width_per_box = int((vp.total_display_width - vp.padding_left - vp.padding_right) / (len(sag.dag_format) + (len(sag.dag_format) - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        elements_of_the_graph = []
        for i, nodes in enumerate(sag.dag_format):
            for j, node in enumerate(nodes):
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
                html.Button('Refresh', id='refresh-button', n_clicks=0),
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
            def create_slider_data_from_list(previous_value, options_list):
                options_list = sorted(options_list)
                val = previous_value if previous_value in options_list else (options_list[0] if options_list else None)
                marks = {a: str(a) for a in options_list}
                min_, max_ = (options_list[0], options_list[-1]) if options_list else (0, 100)
                return min_, max_, marks, val
            def create_options_and_value_from_list(previous_value, options_list, label_maker=lambda a: str(a)):
                val = previous_value if previous_value in options_list else (options_list[0] if options_list else None)
                options = [{'label': label_maker(a), 'value': a} for a in options_list]
                return options, val

            recordings_path = self.path / 'trials' / trials_value / 'recordings'
            recording_files = list(recordings_path.iterdir())
            training_steps = sorted([int(a.stem) for a in recording_files])
            assert len(training_steps) > 0
            loaded_training_steps = sorted(list(recordings.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records.keys()))
            for a in loaded_training_steps:
                assert a in training_steps, (a, training_steps)
            training_step_min, training_step_max, training_step_marks, training_step_value = create_slider_data_from_list(training_step_value, training_steps)
            type_of_recording_to_batch_index_to_iteration_to_role_to_records = recordings.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records[training_step_value]
            types_of_recordings = sorted(list(type_of_recording_to_batch_index_to_iteration_to_role_to_records.keys()))
            assert len(types_of_recordings) > 0
            type_of_recording_options, type_of_recording_value = create_options_and_value_from_list(type_of_recording_value, types_of_recordings)
            batch_index_to_iteration_to_role_to_records = type_of_recording_to_batch_index_to_iteration_to_role_to_records[type_of_recording_value]
            batch_indices = sorted(list(batch_index_to_iteration_to_role_to_records.keys()), key=lambda x: -1 if isinstance(x, str) else x)
            assert len(batch_indices) > 0
            batch_index_options, batch_index_value = create_options_and_value_from_list(
                batch_index_value, batch_indices,
                label_maker=lambda a: "mean over the batch" if a == 'batch' else f"batch index {a}"
            )
            iteration_to_role_to_records = batch_index_to_iteration_to_role_to_records[batch_index_value]
            iteration_steps = sorted(list(iteration_to_role_to_records.keys()))
            assert len(iteration_steps) > 0
            iteration_min, iteration_max, iteration_marks, iteration_value = create_slider_data_from_list(iteration_value, iteration_steps)
            configuration_type = recordings.training_step_to_iteration_to_configuration_type[training_step_value][iteration_value]
            sag = self.configuration_type_to_status_and_graph[configuration_type]
            node = sag.name_to_node.get(name_of_selected_node, None)
            graph_container_visibilities = [
                'active' if conf_type == configuration_type else 'inactive'
                for conf_type in self.configuration_type_to_status_and_graph.keys()
            ]
            role_to_records, value_is_independent_of_iterations = self.pick_iteration_for_nodes_without_iteration_selection(
                iteration_value, node, iteration_to_role_to_records)
            possible_roles = list(dict.fromkeys([
                role
                for role, v in role_to_records.items()
                for (node, _, _) in v.keys()
                if node == name_of_selected_node
            ]))
            role_of_tensor_in_node_options, role_of_tensor_in_node_value = create_options_and_value_from_list(role_of_tensor_in_node_value, possible_roles)
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
                       Input('iteration-slider', 'value')] +
                      [Input(self._node_name_to_dash_id(configuration_type, node), 'n_clicks_timestamp')
                       for configuration_type, sag in self.configuration_type_to_status_and_graph.items()
                       for node in sag.name_to_node.keys()])
        @utilities.runtime_analysis_decorator
        def update_visibility(*lsts):
            trials_value = lsts[0]
            training_step_value = lsts[1]
            iteration_value = lsts[2]
            clicks_per_object = lsts[3:]
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
            assert len(clicks_per_object) == len(names)
            # Identify the most recently clicked object
            max_index = [0 if a is None else 1 for a in names].index(1)
            max_val = 0
            for i, (clicks, name) in enumerate(zip(clicks_per_object, names)):
                if name in names_that_exist_in_the_selected_configuration and clicks is not None and clicks > max_val:
                    max_val = clicks
                    max_index = i
            name_of_selected_node = names[max_index]
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
            batch_index_to_iteration_to_role_to_records = recordings.training_step_to_type_of_recording_to_batch_index_to_iteration_to_role_to_records[training_step_value][type_of_recording_value]
            iteration_to_role_to_records, value_is_independent_of_batch_index = self.pick_batch_index_for_nodes_without_batch_index_selection(
                batch_index_value, node, batch_index_to_iteration_to_role_to_records)
            role_to_records, value_is_independent_of_iterations = self.pick_iteration_for_nodes_without_iteration_selection(
                iteration_value, node, iteration_to_role_to_records)
            if role_of_tensor_in_node_value is None:
                possible_roles = list(dict.fromkeys([
                    role
                    for role, v in role_to_records.items()
                    for (n, _, _) in v.keys()
                    if n == node.full_unique_name
                ]))
                assert len(possible_roles) == 1, (len(possible_roles), node.full_unique_name)
                role_of_tensor_in_node_value = possible_roles[0]
            records = role_to_records[role_of_tensor_in_node_value]
            rows = []
            for (nn, item, metadata), val in records.items():
                if nn != node_name:
                    continue
                row = [
                    html.Td(item),
                    html.Td(metadata),
                    html.Td(val),
                ]
                rows.append(html.Tr(row))
            desc_text = node.type_of_tensor
            if value_is_independent_of_batch_index:
                desc_text += "  -  NOTE: Values are independent of the selected batch index. Displaying values for no selected index."
            if value_is_independent_of_iterations:
                desc_text += "  -  NOTE: Values are independent of the selected iteration. Displaying values for iteration 0."
            tensor_shape = recordings.node_to_role_to_tensor_metadata[node.full_unique_name][role_of_tensor_in_node_value].shape
            children = [
                html.Header(f"{node.full_unique_name} - {role_of_tensor_in_node_value}"),
                html.P(desc_text),
                html.P(f"Shape: [{', '.join([str(a) for a in tensor_shape])}]"),
                html.Table([html.Tr([html.Th(col) for col in ['KPI', 'metadata', 'value']])] + rows)
            ]
            return children, graph_overlay_for_selections_children

    def pick_batch_index_for_nodes_without_batch_index_selection(self, batch_index_value, node, batch_index_to_iteration_to_role_to_records):
        if node is not None and node.type_of_tensor == 'parameter':
            iteration_to_role_to_records = batch_index_to_iteration_to_role_to_records['batch']
            value_is_independent_of_batch_indices = True
            for batch_index_, iteration_to_role_to_records_ in batch_index_to_iteration_to_role_to_records.items():
                for iteration, role_to_records_ in iteration_to_role_to_records_.items():
                    for records_ in role_to_records_.values():
                        for nn, _, _ in records_.keys():
                            if nn == node.full_unique_name:
                                assert batch_index_ == 'batch', \
                                    f"Should have an entry only for no batch_index.\n" \
                                    f"{batch_index_}\n{node.full_unique_name}"
        else:
            iteration_to_role_to_records = batch_index_to_iteration_to_role_to_records[batch_index_value]
            value_is_independent_of_batch_indices = False
        return iteration_to_role_to_records, value_is_independent_of_batch_indices

    def pick_iteration_for_nodes_without_iteration_selection(self, iteration_value, node, iteration_to_role_to_records):
        if node is not None and node.type_of_tensor == 'parameter':
            role_to_records = iteration_to_role_to_records[0]
            value_is_independent_of_iterations = True
            for iteration, role_to_records_ in iteration_to_role_to_records.items():
                for records_ in role_to_records_.values():
                    for nn, _, _ in records_.keys():
                        if nn == node.full_unique_name:
                            assert iteration == 0, \
                                f"Should have an entry only for iteration 0.\n" \
                                f"{iteration}\n{node.full_unique_name}"
        else:
            role_to_records = iteration_to_role_to_records[iteration_value]
            value_is_independent_of_iterations = False
        return role_to_records, value_is_independent_of_iterations
