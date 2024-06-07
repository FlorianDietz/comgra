# BASIC IMPORTS
import collections
import importlib
import itertools
import os.path
from dataclasses import dataclass
import gzip
import json
import math
import numbers
from pathlib import Path
import pickle
import re
import threading
from typing import Dict, List, Tuple

import torch
import dash
from dash import ctx, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import dash_svg
import msgpack

from comgra.objects import TrainingStepConfiguration, NodeGraphStructure
from comgra.utilities import PseudoDb

from typing import Any, Dict, Generic, List, NamedTuple, Optional, Tuple, TYPE_CHECKING, Union


def create_visualization(
        recordings, type_of_execution, tsc: TrainingStepConfiguration, ngs: NodeGraphStructure, db: PseudoDb,
        training_step_value, type_of_recording_value, batch_index_value, iteration_value,
        node_name, role_of_tensor_in_node_value,
):
    messages = [f"This is a demonstration of custom visualization, using the file 'example_custom_visualization.py'.\n"]

    filters = {
        # Note: The 'recordings' are not used in the filters because comgra uses a separate database object
        # for each recording, so this filter is already covered implicitly
        'training_step': training_step_value,
        'type_of_tensor_recording': type_of_recording_value,
        'batch_aggregation': batch_index_value,
        'iteration': iteration_value,
        'node_name': node_name,
        'role_within_node': role_of_tensor_in_node_value,
        'record_type': 'data',
        # 'item': None,
        # 'metadata': None,
    }
    list_of_matches, possible_attribute_values = db.get_matches(filters)

    messages.append(f"filters:\n{filters}\n")

    messages.append(f"number of matches: {len(list_of_matches)}:\n")
    for match in list_of_matches:
        messages.append(f"{match}\n")
    messages.append(f"possible_attribute_values:\n{possible_attribute_values}\n")

    # Comgra uses https://github.com/plotly/dash for visualization.
    # This function needs to return a dash element.
    my_div = html.Div(
        messages,
        style={
            'white-space': 'pre-wrap',
        }
    )
    return html.Div(id="custom_visualization", className='', style={
            'display': 'flex',
            'flex-wrap': 'wrap',
        }, children=my_div)
