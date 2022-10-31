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

from comgra.objects import DirectedAcyclicGraph, GlobalStatus, ModuleRepresentation, ParameterRepresentation, TensorRepresentation


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
        self._sanity_check_cache_to_avoid_duplicates = {}

    def _node_name_to_dash_id(self, node_name):
        conversion = node_name.replace('.', '__')
        assert (conversion not in self._sanity_check_cache_to_avoid_duplicates or
               self._sanity_check_cache_to_avoid_duplicates[conversion] == node_name), \
            f"Two nodes have the same representation: " \
            f"{node_name}, {self._sanity_check_cache_to_avoid_duplicates[conversion]}, {conversion}"
        self._sanity_check_cache_to_avoid_duplicates[conversion] = node_name
        return conversion

    def _nodes_to_connection_dash_id(self, source, target):
        return f"connection__{self._node_name_to_dash_id(source)}__{self._node_name_to_dash_id(target)}"

    def run_server(self):
        self.create_visualization()
        self.app.run_server(debug=True)

    def create_nodes_and_arrows(self, global_status: GlobalStatus, graph: DirectedAcyclicGraph):
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
                elements_of_the_graph.append(html.Div(id=self._node_name_to_dash_id(node), style={
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
            connection_name = self._nodes_to_connection_dash_id(source, target)
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
        return elements_of_the_graph, connection_names_to_source_and_target, node_to_incoming_and_outgoing_lines

    def create_visualization(self):
        app = self.app

        with open(self.path / 'globals.json') as f:
            globals_json = json.load(f)
            globals_json['tensor_representations'] = {k: TensorRepresentation(**v) for k, v in globals_json['tensor_representations'].items()}
            global_status = GlobalStatus(**globals_json)
        with open(self.path / 'graph.json') as f:
            graph_json = json.load(f)
            graph = DirectedAcyclicGraph(**graph_json)
            print(graph.dag_format)

        elements_of_the_graph, connection_names_to_source_and_target, node_to_incoming_and_outgoing_lines = \
            self.create_nodes_and_arrows(global_status, graph)

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
            }, children=elements_of_the_graph),
            dcc.Tooltip(id="graph-tooltip"),
            html.Div(id='controls-container', children=[
                html.Button('Refresh', id='refresh-button', n_clicks=0),
                dcc.Dropdown(
                    id='trials-dropdown',
                    options=[],
                    value=None,
                ),
            ]),
            html.Div(id='selected-item-details-container', children=[
            ]),
        ])

        @app.callback([Output('trials-dropdown', 'options'),
                       Output('trials-dropdown', 'value')],
                      [Input('refresh-button', 'n_clicks')])
        def update_trials_list(n_clicks):
            trials_folder = self.path / 'trials'
            subfolders = [a for a in trials_folder.iterdir() if a.is_dir()]
            options = [{'label': a.name, 'value': a.name} for a in subfolders]
            return options, options[0]['value']
        update_trials_list(0)

        @app.callback(Output('dummy-for-selecting-a-node', 'className'),
                      [Input(self._node_name_to_dash_id(node), 'n_clicks_timestamp') for node
                       in global_status.tensor_representations.keys()])
        def update_visibility(*lsts):
            num_nodes = len(global_status.tensor_representations)
            clicks_per_object = lsts[0:num_nodes]
            # Identify the most recently clicked object
            max_index = 0
            max_val = 0
            for i, b in enumerate(clicks_per_object):
                if b is not None and b > max_val:
                    max_val = b
                    max_index = i
            name_of_selected_item = list(global_status.tensor_representations.keys())[max_index]
            return name_of_selected_item

        @app.callback(
            ([Output('selected-item-details-container', 'children')] +
             [Output(self._node_name_to_dash_id(n), 'className') for n in global_status.tensor_representations]),
            Input('dummy-for-selecting-a-node', 'className')
        )
        def select_node(node_name):
            node = global_status.tensor_representations[node_name]
            children = [
                html.Header(node.full_unique_name),
                html.P(node.role),
                html.P(f"[{', '.join([str(a) for a in node.shape])}]"),
            ]
            connected_node_names = {a[0] for a in graph.connections if a[1] == node_name} | {a[1] for a in graph.connections if a[0] == node_name}
            classes_for_nodes = [
                "node selected" if n == node_name else "node highlighted" if n in connected_node_names else "node"
                for n in global_status.tensor_representations
            ]
            return [children] + classes_for_nodes
