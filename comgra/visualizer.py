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
        self.app.run_server()

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
                html.Div(id='main_area', style={
                    'width': f'100px',
                    'height': f'100px',
                    'backgroundColor': 'white',
                    'position': 'absolute',
                    'left': '50px',
                    'top': '100px',
                    'border': '1px solid black',
                }),
            ]),
            html.Div(id='controls_container'),
        ])
        # TODO
        #  -a function that updates the graph display, to be triggered whenever a node is expanded or closed
        #  -it always creates a left-to-right layout and adjusts all heights and widths dynamically to fit the whole size
        #  -the layout is: left to right. Always alternating between tensors and modules. --> How do I make this as compact as possible?
        #    have a look at a practical example, first.
