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

from comgra.objects import DirectedAcyclicGraph, GlobalStatus, ModuleRepresentation, ParameterRepresentation, TensorRepresentation, TensorRecordings


@dataclass
class VisualizationParameters:
    total_display_width = 1800
    total_display_height = 600
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
    color_for_highlighting_primary = '#ff1700'
    color_for_highlighting_secondary = '#db8468'


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
        self.configuration_type_to_graph_and_status = {}
        self._sanity_check_cache_to_avoid_duplicates = {}
        self.cache_for_tensor_recordings = {}

    def _node_name_to_dash_id(self, configuration_type, node_name):
        conversion = node_name.replace('.', '__')
        conversion = f"node__{configuration_type}__{conversion}"
        assert (conversion not in self._sanity_check_cache_to_avoid_duplicates or
               self._sanity_check_cache_to_avoid_duplicates[conversion] == node_name), \
            f"Two nodes have the same representation: " \
            f"{node_name}, {self._sanity_check_cache_to_avoid_duplicates[conversion]}, {conversion}"
        self._sanity_check_cache_to_avoid_duplicates[conversion] = node_name
        return conversion

    def _nodes_to_connection_dash_id(self, configuration_type, source, target):
        return f"connection__{self._node_name_to_dash_id(configuration_type, source)}__" \
               f"{self._node_name_to_dash_id(configuration_type, target)}"

    def run_server(self):
        #
        # Load data that can only be loaded once, because Divs depend on it.
        #
        for config_folder in (self.path / 'configs').iterdir():
            configuration_type = config_folder.name
            with open(config_folder / 'globals.json') as f:
                globals_json = json.load(f)
                globals_json['tensor_representations'] = {
                    k: TensorRepresentation(**v) for k, v in
                    globals_json['tensor_representations'].items()
                }
                global_status = GlobalStatus(**globals_json)
            with open(config_folder / 'graph.json') as f:
                graph_json = json.load(f)
                graph = DirectedAcyclicGraph(**graph_json)
            self.configuration_type_to_graph_and_status[configuration_type] = (graph, global_status)
        #
        # Visualize
        #
        self.create_visualization()
        self.app.run_server(debug=True)

    def get_recordings_with_caching(self, trials_value) -> TensorRecordings:
        key = (trials_value,)
        if key not in self.cache_for_tensor_recordings:
            recordings_file = self.path / 'trials' / trials_value / 'recordings.json'
            with open(recordings_file, 'rb') as f:
                recordings: TensorRecordings = pickle.load(f)
            self.cache_for_tensor_recordings[key] = recordings
        return self.cache_for_tensor_recordings[key]

    def create_nodes_and_arrows(
            self, configuration_type: str, global_status: GlobalStatus, graph: DirectedAcyclicGraph
    ):
        highest_number_of_nodes = max(len(a) for a in graph.dag_format)
        height_per_box = int((vp.total_display_height - vp.padding_top - vp.padding_bottom) / (highest_number_of_nodes + (highest_number_of_nodes - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        width_per_box = int((vp.total_display_width - vp.padding_left - vp.padding_right) / (len(graph.dag_format) + (len(graph.dag_format) - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        elements_of_the_graph = []
        node_to_top_left_corner = {}
        for i, nodes in enumerate(graph.dag_format):
            for j, node in enumerate(nodes):
                left = int(vp.padding_left + i * (1 + vp.ratio_of_space_between_nodes_to_node_size) * width_per_box)
                top = int(vp.padding_top + j * (1 + vp.ratio_of_space_between_nodes_to_node_size) * height_per_box)
                node_to_top_left_corner[node] = (left, top)
                elements_of_the_graph.append(html.Div(id=self._node_name_to_dash_id(configuration_type, node), style={
                    'width': f'{width_per_box}px',
                    'height': f'{height_per_box}px',
                    'backgroundColor': 'white',
                    'position': 'absolute',
                    'left': f'{left}px',
                    'top': f'{top}px',
                    'background': vp.node_type_to_color[global_status.tensor_representations[node].role]
                }, title=node, children=[
                    f'{node}'
                ]))
        connection_names_to_source_and_target = {}
        node_to_incoming_and_outgoing_lines = {n: ([], []) for n in graph.nodes}
        svg_connection_lines = []
        for connection in graph.connections:
            source, target = tuple(connection)
            source_left, source_top = node_to_top_left_corner[source]
            target_left, target_top = node_to_top_left_corner[target]
            source_x = int(source_left + width_per_box)
            source_y = int(source_top + 0.5 * height_per_box)
            target_x = int(target_left)
            target_y = int(target_top + 0.5 * height_per_box)
            connection_name = self._nodes_to_connection_dash_id(configuration_type, source, target)
            svg_connection_lines.append(dash_svg.Line(
                id=connection_name,
                x1=str(source_x), x2=str(target_x), y1=str(source_y), y2=str(target_y),
                stroke=vp.node_type_to_color[global_status.tensor_representations[source].role],
                strokeWidth=1,
            ))
            connection_names_to_source_and_target[connection_name] = (source, target)
            node_to_incoming_and_outgoing_lines[source][1].append(connection_name)
            node_to_incoming_and_outgoing_lines[target][0].append(connection_name)
        # Note: "9" here is the offset of the viewport from the top left corner of the screen.
        # This is 1 from the border plus 8 from the padding.
        elements_of_the_graph.append(dash_svg.Svg(svg_connection_lines, viewBox=f'9 9 {vp.total_display_width} {vp.total_display_height}'))
        return elements_of_the_graph

    def create_visualization(self):
        app = self.app
        # Define the Layout of the App
        app.layout = html.Div(style={
            'backgroundColor': 'white',
        }, children=[
            html.Div(id='dummy-for-selecting-a-node'),  # This dummy stores some data
            html.Div(id='dummy-for-hovering-on-a-node'),  # This dummy stores some data
            html.Div(id='graph-container', style={
                'display': 'relative',
                'width': f'{vp.total_display_width}px',
                'height': f'{vp.total_display_height}px',
                'border': '1px solid black',
            }, children=[
                html.Div(id=f'graph_container__{configuration_type}', style={
                    'display': 'relative',
                    'width': f'{vp.total_display_width}px',
                    'height': f'{vp.total_display_height}px',
                }, children=self.create_nodes_and_arrows(configuration_type, global_status, graph))
                for configuration_type, (graph, global_status) in self.configuration_type_to_graph_and_status.items()
            ]),
            dcc.Tooltip(id="graph-tooltip"),
            html.Div(id='controls-container', children=[
                html.Button('Refresh', id='refresh-button', n_clicks=0),
                dcc.Dropdown(id='trials-dropdown', options=[], value=None),
                dcc.Slider(id='training-step-slider', min=0, max=100, step=None, value=None),
                dcc.RadioItems(id='type-of-recording-dropdown', options=[], value=None),
                dcc.Dropdown(id='batch-index-dropdown', options=[], value=None),
            ]),
            html.Div(id='selected-item-details-container', children=[
            ]),
        ])

        @app.callback([Output('trials-dropdown', 'options'),
                       Output('trials-dropdown', 'value')],
                      [Input('refresh-button', 'n_clicks')])
        def refresh_all(n_clicks):
            # Reset caches
            self.cache_for_tensor_recordings = {}
            # Load the list of trials
            trials_folder = self.path / 'trials'
            subfolders = [a for a in trials_folder.iterdir() if a.is_dir()]
            options = [{'label': a.name, 'value': a.name} for a in subfolders]
            return options, options[0]['value']
        refresh_all(0)

        @app.callback(([Output('training-step-slider', 'min'),
                       Output('training-step-slider', 'max'),
                       Output('training-step-slider', 'marks'),
                       Output('training-step-slider', 'value'),
                       Output('type-of-recording-dropdown', 'options'),
                       Output('type-of-recording-dropdown', 'value'),
                       Output('batch-index-dropdown', 'options'),
                       Output('batch-index-dropdown', 'value')] +
                      [Output(f'graph_container__{configuration_type}', 'className')
                       for configuration_type in self.configuration_type_to_graph_and_status.keys()]),
                      [Input('trials-dropdown', 'value'),
                       Input('training-step-slider', 'value'),
                       Input('type-of-recording-dropdown', 'value'),
                       Input('batch-index-dropdown', 'value')])
        def update_dropboxes(trials_value, training_step_value, type_of_recording_value, batch_index_value):
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
            training_steps = sorted(list(recordings.training_step_to_type_of_recording_to_batch_index_to_records.keys()))
            assert len(training_steps) > 0
            training_step_min, training_step_max, training_step_marks, training_step_value = create_slider_data_from_list(training_step_value, training_steps)
            type_of_recording_to_batch_index_to_records = recordings.training_step_to_type_of_recording_to_batch_index_to_records[training_step_value]
            types_of_recordings = sorted(list(type_of_recording_to_batch_index_to_records.keys()))
            assert len(types_of_recordings) > 0
            type_of_recording_options, type_of_recording_value = create_options_and_value_from_list(type_of_recording_value, types_of_recordings)
            batch_index_to_records = type_of_recording_to_batch_index_to_records[type_of_recording_value]
            batch_indices = sorted(list(batch_index_to_records.keys()), key=lambda x: -1 if isinstance(x, str) else x)
            assert len(batch_indices) > 0
            batch_index_options, batch_index_value = create_options_and_value_from_list(
                batch_index_value, batch_indices,
                label_maker=lambda a: "mean over the batch" if a == 'batch' else f"batch index {a}"
            )
            graph_container_visibilities = [
                'active' if configuration_type == recordings.configuration_type else 'inactive'
                for configuration_type in self.configuration_type_to_graph_and_status.keys()
            ]
            res = [training_step_min, training_step_max, training_step_marks, training_step_value, type_of_recording_options, type_of_recording_value, batch_index_options, batch_index_value] + graph_container_visibilities
            return res

        @app.callback(Output('dummy-for-selecting-a-node', 'className'),
                      [Input('trials-dropdown', 'value')] +
                      [Input(self._node_name_to_dash_id(configuration_type, node), 'n_clicks_timestamp')
                       for configuration_type, (_, global_status) in self.configuration_type_to_graph_and_status.items()
                       for node in global_status.tensor_representations.keys()])
        def update_visibility(*lsts):
            trials_value = lsts[0]
            recordings = self.get_recordings_with_caching(trials_value)
            graph, global_status = self.configuration_type_to_graph_and_status[recordings.configuration_type]
            names = [
                node
                for _, (_, gs) in self.configuration_type_to_graph_and_status.items()
                for node in gs.tensor_representations.keys()
            ]
            names_that_exist_in_the_selected_configuration = {
                node for node in
                self.configuration_type_to_graph_and_status[recordings.configuration_type][1].tensor_representations.keys()
            }
            clicks_per_object = lsts[1:]
            assert len(clicks_per_object) == len(names)
            # Identify the most recently clicked object
            max_index = [0 if a is None else 1 for a in names].index(1)
            max_val = 0
            for i, (clicks, name) in enumerate(zip(clicks_per_object, names)):
                if name in names_that_exist_in_the_selected_configuration and clicks is not None and clicks > max_val:
                    max_val = clicks
                    max_index = i
            name_of_selected_item = names[max_index]
            assert name_of_selected_item is not None
            return name_of_selected_item

        @app.callback(
            ([Output('selected-item-details-container', 'children')] +
             [Output(self._node_name_to_dash_id(configuration_type, n), 'className')
              for configuration_type, (_, global_status) in self.configuration_type_to_graph_and_status.items()
              for n in global_status.tensor_representations]),
            [Input('dummy-for-selecting-a-node', 'className'),
             Input('trials-dropdown', 'value'),
             Input('training-step-slider', 'value'),
             Input('type-of-recording-dropdown', 'value'),
             Input('batch-index-dropdown', 'value')]
        )
        def select_node(node_name, trials_value, training_step_value, type_of_recording_value, batch_index_value):
            recordings = self.get_recordings_with_caching(trials_value)
            graph, global_status = self.configuration_type_to_graph_and_status[recordings.configuration_type]
            # Select the node
            node = global_status.tensor_representations[node_name]
            connected_node_names = {a[0] for a in graph.connections if a[1] == node_name} | {a[1] for a in graph.connections if a[0] == node_name}
            classes_for_nodes = [
                ("node selected" if n == node_name else "node highlighted" if n in connected_node_names else "node")
                if gs == global_status else "node inactive"
                for _, (_, gs) in self.configuration_type_to_graph_and_status.items()
                for n in gs.tensor_representations
            ]
            # Display values based on the selected node
            records = recordings.training_step_to_type_of_recording_to_batch_index_to_records[training_step_value][type_of_recording_value][batch_index_value]
            rows = []
            for key in node.get_all_items_to_record():
                val = records[key] if key in records else "No applicable value to display for this selection."
                assert key[0] == node_name
                row = [
                    html.Td(key[1]),
                    html.Td(key[2]),
                    html.Td(val),
                ]
                rows.append(html.Tr(row))
            children = [
                html.Header(node.full_unique_name),
                html.P(node.role),
                html.P(f"[{', '.join([str(a) for a in node.shape])}]"),
                html.Table([html.Tr([html.Th(col) for col in ['KPI', 'metadata', 'value']])] + rows)
            ]
            return [children] + classes_for_nodes
