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
from dash import dcc, html
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

    def run_server(self):
        self.create_visualization()
        self.app.run_server(debug=True)

    def create_visualization(self):
        app = self.app

        # Define the Layout of the App
        app.layout = html.Div(style={
            'backgroundColor': 'white',
        }, children=[
            html.Div(id='graph_container', style={
                'display': 'relative',
                'width': f'{vp.total_display_width}px',
                'height': f'{vp.total_display_height}px',
                'border': '1px solid black',
            }, children=[

            ]),
            html.Div(id='controls_container', children=[
                html.Button('Refresh', id='refresh-button', n_clicks=0),
            ]),
        ])

        @app.callback(
            Output('graph_container', 'children'),
            Input('refresh-button', 'n_clicks'),
        )
        def reload_graph(n_clicks):
            with open(self.path / 'globals.json') as f:
                globals_json = json.load(f)
                globals_json['tensor_representations'] = [TensorRepresentation(**a) for a in globals_json['tensor_representations']]
                global_status = GlobalStatus(**globals_json)
            with open(self.path / 'graph.json') as f:
                graph_json = json.load(f)
                graph = DirectedAcyclicGraph(**graph_json)
                print(graph.dag_format)
            highest_number_of_nodes = max(len(a) for a in graph.dag_format)
            height_per_box = int((vp.total_display_height - vp.padding_top - vp.padding_bottom) / (highest_number_of_nodes + (highest_number_of_nodes - 1) * vp.ratio_of_space_between_nodes_to_node_size))
            width_per_box = int((vp.total_display_width - vp.padding_left - vp.padding_right) / (len(graph.dag_format) + (len(graph.dag_format) - 1) * vp.ratio_of_space_between_nodes_to_node_size))
            children = []
            for i, nodes in enumerate(graph.dag_format):
                for j, node in enumerate(nodes):
                    children.append(html.Div(id='main_area', style={
                        'width': f'{width_per_box}px',
                        'height': f'{height_per_box}px',
                        'backgroundColor': 'white',
                        'position': 'absolute',
                        'left': f'{int(vp.padding_left + i * (1 + vp.ratio_of_space_between_nodes_to_node_size) * width_per_box)}px',
                        'top': f'{int(vp.padding_top + j * (1 + vp.ratio_of_space_between_nodes_to_node_size) * height_per_box)}px',
                        'border': '1px solid black',
                    }, children=[
                        f'{node}'
                    ]))
            # TODO
            #  *make the nodes have a tooltip
            #  *color coding
            #  *details that show up when you select a node
            #  *selecting a node OR mouseover on it highlights the node as well as all connections to and from it
            return children
        # TODO
        #  -think about how to make the recording work concurrently with many trials in parallel AND in sequence
        #    note that the graph in particular will end up having multiple different variants depending on parameters
        #    it should be possible to explicitly list which variant of the graph a particular trial run is recording.
        #    then it will run sanity checks comparing to that version, but not other versions.
