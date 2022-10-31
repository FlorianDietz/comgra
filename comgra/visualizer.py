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


vp = VisualizationParameters()


class Visualization:
    """
    Keeps track of KPIs over the course of the Experiment and creates visualizations from them.
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        assert path.exists(), path
        self.app = dash.Dash(__name__)
        self._sanity_check_cache_to_avoid_duplicates = {}

    def _node_name_to_dash_id(self, node_name):
        conversion = node_name.replace('.', '__')
        assert (conversion not in self._sanity_check_cache_to_avoid_duplicates or
               self._sanity_check_cache_to_avoid_duplicates[conversion] == node_name), \
            f"Two nodes have the same representation: " \
            f"{node_name}, {self._sanity_check_cache_to_avoid_duplicates[conversion]}, {conversion}"
        self._sanity_check_cache_to_avoid_duplicates[conversion] = node_name
        return conversion

    def run_server(self):
        self.create_visualization()
        self.app.run_server(debug=True)

    def create_nodes(self, global_status: GlobalStatus, graph: DirectedAcyclicGraph):
        highest_number_of_nodes = max(len(a) for a in graph.dag_format)
        height_per_box = int((vp.total_display_height - vp.padding_top - vp.padding_bottom) / (highest_number_of_nodes + (highest_number_of_nodes - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        width_per_box = int((vp.total_display_width - vp.padding_left - vp.padding_right) / (len(graph.dag_format) + (len(graph.dag_format) - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        elements_of_the_graph = []
        for i, nodes in enumerate(graph.dag_format):
            for j, node in enumerate(nodes):
                elements_of_the_graph.append(html.Div(id=self._node_name_to_dash_id(node), style={
                    'title': node,
                    'width': f'{width_per_box}px',
                    'height': f'{height_per_box}px',
                    'backgroundColor': 'white',
                    'position': 'absolute',
                    'left': f'{int(vp.padding_left + i * (1 + vp.ratio_of_space_between_nodes_to_node_size) * width_per_box)}px',
                    'top': f'{int(vp.padding_top + j * (1 + vp.ratio_of_space_between_nodes_to_node_size) * height_per_box)}px',
                    'border': '1px solid black',
                    'background': vp.node_type_to_color[global_status.tensor_representations[node].role]
                }, title=node, children=[
                    f'{node}'
                ]))
        return elements_of_the_graph

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

        elements_of_the_graph = self.create_nodes(global_status, graph)

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
            ]),
            html.Div(id='selected-item-details-container', children=[
            ]),
        ])

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
            Output('selected-item-details-container', 'children'),
            Input('dummy-for-selecting-a-node', 'className')
        )
        def select_node(node_name):
            node = global_status.tensor_representations[node_name]
            children = [
                html.Header(node.full_unique_name),
                html.P(node.role),
                html.P(f"[{', '.join([str(a) for a in node.shape])}]"),
            ]
            return children
        # TODO
        #  *selecting a node OR mouseover on it highlights the node as well as all connections to and from it
        # TODO
        #  -think about how to make the recording work concurrently with many trials in parallel AND in sequence
        #    note that the graph in particular will end up having multiple different variants depending on parameters
        #    it should be possible to explicitly list which variant of the graph a particular trial run is recording.
        #    then it will run sanity checks comparing to that version, but not other versions.
