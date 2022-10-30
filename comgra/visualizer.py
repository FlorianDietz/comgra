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

from comgra.objects import DirectedAcyclicGraph, ModuleRepresentation, ParameterRepresentation, TensorRepresentation


@dataclass
class VisualizationParameters:
    total_display_width = 1800
    total_display_height = 600


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
            with open(self.path / 'graph.json') as f:
                graph_json = json.load(f)
                print(graph_json)
                graph = DirectedAcyclicGraph(**graph_json)
                print(graph.build_dag_format())
            children = [html.Div(id='main_area', style={
                    'width': f'100px',
                    'height': f'100px',
                    'backgroundColor': 'white',
                    'position': 'absolute',
                    'left': '50px',
                    'top': '100px',
                    'border': '1px solid black',
                })]
            return children
        # TODO
        #  -it always creates a left-to-right layout and adjusts all heights and widths dynamically to fit the whole size
        #  -note that modules & tensors can be nested within other modules.
        #    should collapsing a module always hide all tensors inside it,
        #    or can it be useful to investigate the hidden ones some times?
        #  -the layout is: left to right. Always alternating between tensors and modules.
        #  --> How do I make this as compact as possible?
        #    have a look at a practical example, first.
        #    also google for "graph layouting logic"
        #  -think about how to make the recording work concurrently with many trials in parallel AND in sequence
        #    note that the graph in particular will end up having multiple different variants depending on parameters
        #    it should be possible to explicitly list which variant of the graph a particular trial run is recording.
        #    then it will run sanity checks comparing to that version, but not other versions.
