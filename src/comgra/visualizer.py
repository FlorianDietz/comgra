# BASIC IMPORTS
import collections
import importlib
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

import dash
from dash import ctx, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_svg
import msgpack
import plotly.graph_objs as go

from comgra.objects import StatusAndGraph, ModuleRepresentation, TensorRecordings
from comgra import utilities

DISPLAY_ALL_CONNECTIONS_GRAPHICALLY = False
HIGHLIGHT_SELECTED_CONNECTIONS = True
DISPLAY_NAMES_ON_NODES_GRAPHICALLY = True

LOCK_FOR_RECORDINGS = threading.Lock()


@dataclass
class VisualizationParameters:
    total_display_width = 1800
    total_display_height = 400
    padding_left = 50
    padding_right = 50
    padding_top = 50
    padding_bottom = 50
    ratio_of_space_between_nodes_to_node_size = 1.0
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


class CustomDash(dash.Dash):
    def interpolate_index(self, **kwargs):
        return '''
<!DOCTYPE html>
<html>
    <head>
        {metas}
        <title>{title}</title>

        <link rel="apple-touch-icon" sizes="180x180" href="assets/favicons/apple-touch-icon.png">
        <link rel="icon" type="image/png" sizes="32x32" href="assets/favicons/favicon-32x32.png">
        <link rel="icon" type="image/png" sizes="16x16" href="assets/favicons/favicon-16x16.png">
        <link rel="manifest" href="assets/favicons/site.webmanifest">
        <link rel="mask-icon" href="assets/favicons/safari-pinned-tab.svg" color="#5bbad5">
        <meta name="msapplication-TileColor" content="#da532c">
        <meta name="theme-color" content="#ffffff">

        {css}
    </head>
    <body>
        {app_entry}
        <footer>
            {config}
            {scripts}
            {renderer}
        </footer>
    </body>
</html>
        '''.format(**kwargs)


class Visualization:
    """
    Keeps track of KPIs over the course of the Experiment and creates visualizations from them.
    """

    def __init__(self, path, debug_mode, external_visualization_file):
        super().__init__()
        utilities.DEBUG_MODE = debug_mode
        self.debug_mode = debug_mode
        self.path: Path = path
        self.external_visualization_file = external_visualization_file
        assert path.exists(), path
        assets_path = Path(__file__).absolute().parent.parent / 'assets'
        assert assets_path.exists(), "If this fails, files have been moved."
        self.app = CustomDash(__name__, assets_folder=str(assets_path),
                              external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
        self.configuration_type_to_status_and_graph: Dict[str, StatusAndGraph] = {}
        self.configuration_type_to_node_to_corners: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
        self.configuration_type_to_grid_of_nodes: Dict[str, List[List[str]]] = {}
        self.configuration_type_and_node_to_list_of_connections: Dict[
            Tuple[str, str], List[Tuple[int, int, int, int, str, str]]] = {}
        self._sanity_check_cache_to_avoid_duplicates = {}
        self.cache_for_tensor_recordings = {}
        self.attribute_selection_fallback_values = collections.defaultdict(list)
        self.last_navigation_click_event_time = -1
        self.node_is_a_parameter: Dict[str, bool] = {}
        self.trial_to_kpi_graph_excerpt = {}

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
        # Get helper data
        #
        # Special handling for parameters:
        # They are independent of iterations, and all their values are stored for iteration 0.
        # But we do not want to change the iteration slider when they are selected,
        # and also the layout of the graph may actually change when the iteration changes,
        # so that should be avoided as well.
        all_node_names = set([
            node_name
            for sag in self.configuration_type_to_status_and_graph.values()
            for node_name in sag.name_to_node.keys()
        ])
        for node_name in all_node_names:
            is_parameter_node = set([
                sag.name_to_node[node_name].type_of_tensor == 'parameter'
                for sag in self.configuration_type_to_status_and_graph.values()
                if node_name in sag.name_to_node
            ])
            is_parameter_node = utilities.the(is_parameter_node)
            self.node_is_a_parameter[node_name] = is_parameter_node
        #
        # Visualize
        #
        self.create_visualization()
        self.app.run_server(debug=self.debug_mode, port=port)

    @utilities.runtime_analysis_decorator
    def get_recordings_with_caching(
            self, trials_value, training_step_value, type_of_execution
    ) -> TensorRecordings:
        with LOCK_FOR_RECORDINGS:
            key = (trials_value, training_step_value,)
            recordings = self.cache_for_tensor_recordings.get(key, None)
            if recordings is None:
                recordings_path_base = self.path / 'trials' / trials_value / 'recordings'
                if type_of_execution == 'any_value':
                    for recordings_path in recordings_path_base.iterdir():
                        if recordings_path.name.startswith(f'{training_step_value}__'):
                            break
                else:
                    recordings_path = recordings_path_base / f'{training_step_value}__{type_of_execution}'
                recording_files = sorted(list(recordings_path.iterdir()))
                for recording_file in recording_files:
                    new_recordings_data = self.load_file(recording_file)
                    new_recordings: TensorRecordings = TensorRecordings(**new_recordings_data)
                    new_recordings.recordings = utilities.PseudoDb([]).deserialize(new_recordings.recordings)
                    new_recordings.training_step_to_iteration_to_configuration_type = {
                        int(k1): {
                            int(k2): v2 for k2, v2 in v1.items()
                        }
                        for k1, v1 in new_recordings.training_step_to_iteration_to_configuration_type.items()
                    }
                    new_recordings.training_step_to_type_of_execution = {
                        int(k1): v1
                        for k1, v1 in new_recordings.training_step_to_type_of_execution.items()
                    }
                    if recordings is None:
                        recordings = new_recordings
                    else:
                        recordings.update_with_more_recordings(new_recordings)
                recordings.recordings.create_index(
                    ['training_step', 'record_type', 'node_name', 'role_within_node', 'batch_aggregation', 'iteration'])
                self.cache_for_tensor_recordings[key] = recordings
            return recordings

    @utilities.runtime_analysis_decorator
    def load_file(self, recording_file):
        if recording_file.suffix == '.json':
            new_recordings_data = self.load_json(recording_file)
        elif recording_file.suffix == '.zip_json':
            new_recordings_data = self.load_zip_json(recording_file)
        elif recording_file.suffix == '.pkl':
            new_recordings_data = self.load_pickle(recording_file)
        elif recording_file.suffix == '.msgpack':
            new_recordings_data = self.load_msgpack(recording_file)
        elif recording_file.suffix == '.zip_msgpack':
            new_recordings_data = self.load_zip_msgpack(recording_file)
        else:
            raise ValueError(recording_file)
        return new_recordings_data

    @utilities.runtime_analysis_decorator
    def load_json(self, recording_file):
        with open(recording_file, 'r') as f:
            return json.load(f)

    @utilities.runtime_analysis_decorator
    def load_zip_json(self, recording_file):
        with gzip.open(recording_file, 'r') as f:
            return json.loads(f.read().decode('utf-8'))

    @utilities.runtime_analysis_decorator
    def load_pickle(self, recording_file):
        with open(recording_file, 'rb') as f:
            return pickle.load(f)

    @utilities.runtime_analysis_decorator
    def load_msgpack(self, recording_file):
        with open(recording_file, 'rb') as f:
            return msgpack.load(f, strict_map_key=False)

    @utilities.runtime_analysis_decorator
    def load_zip_msgpack(self, recording_file):
        with gzip.open(recording_file, 'r') as f:
            return msgpack.loads(f.read(), strict_map_key=False)

    @utilities.runtime_analysis_decorator
    def create_nodes_and_arrows(
            self, configuration_type: str, sag: StatusAndGraph
    ) -> List:
        assert configuration_type not in self.configuration_type_to_node_to_corners, configuration_type
        node_to_corners = self.configuration_type_to_node_to_corners.setdefault(configuration_type, {})
        grid_of_nodes = self.configuration_type_to_grid_of_nodes.setdefault(configuration_type, [])
        highest_number_of_nodes = max(len(a) for a in sag.dag_format)
        height_per_box = int((vp.total_display_height - vp.padding_top - vp.padding_bottom) / (
                highest_number_of_nodes + (
                highest_number_of_nodes - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        width_per_box = int((vp.total_display_width - vp.padding_left - vp.padding_right) / (
                len(sag.dag_format) + (len(sag.dag_format) - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        elements_of_the_graph = []
        for i, nodes in enumerate(sag.dag_format):
            list_of_nodes_for_grid = []
            grid_of_nodes.append(list_of_nodes_for_grid)
            left = int(vp.padding_left + i * (1 + vp.ratio_of_space_between_nodes_to_node_size) * width_per_box)
            right = int(left + width_per_box)
            common_prefix = os.path.commonprefix(nodes)
            if '.' in common_prefix:
                common_prefix = common_prefix[:common_prefix.rindex('.') + 1]
            else:
                common_prefix = 'node__'

            def get_appropriate_font_size_for_text_in_node(width, text):
                # TODO This formula was determined experimentally to be "good enough".
                #  Replace it with a better, CSS-based solution.
                return max(1, min(20, int(width / len(text) * 1.7)))

            if common_prefix != 'node__':
                # Display the prefix common to the names of all items in this stack
                top = 0
                text_in_node = common_prefix[len('node__'):-1]
                appropriate_font_size_for_text_in_node = get_appropriate_font_size_for_text_in_node(width_per_box,
                                                                                                    text_in_node)
                elements_of_the_graph.append(
                    html.Div(html.Div(f"{text_in_node}", className='node-name', style={
                        'font-size': f'{appropriate_font_size_for_text_in_node}px'
                    }), className='node dummy-node', style={
                        'width': f'{width_per_box}px',
                        'height': f'{height_per_box}px',
                        'left': f'{left}px',
                        'top': f'{top}px',
                    })
                )
            # Display each of the nodes
            for j, node in enumerate(nodes):
                list_of_nodes_for_grid.append(node)
                top = int(vp.padding_top + j * (1 + vp.ratio_of_space_between_nodes_to_node_size) * height_per_box)
                bottom = int(top + height_per_box)
                node_to_corners[node] = (left, top, right, bottom)
                node_type = sag.name_to_node[node].type_of_tensor
                text_in_node = node[len(common_prefix):]
                appropriate_font_size_for_text_in_node = get_appropriate_font_size_for_text_in_node(width_per_box,
                                                                                                    text_in_node)
                elements_of_the_graph.append(
                    html.Div(id=self._node_name_to_dash_id(configuration_type, node), className='node', style={
                        'width': f'{width_per_box}px',
                        'height': f'{height_per_box}px',
                        'left': f'{left}px',
                        'top': f'{top}px',
                        'background': vp.node_type_to_color[node_type],
                    }, title=node[len("node__"):], children=(
                        [html.Div(f'{text_in_node}', className='node-name', style={
                            'font-size': f'{appropriate_font_size_for_text_in_node}px'
                        })] if DISPLAY_NAMES_ON_NODES_GRAPHICALLY else [])
                             ))
        svg_connection_lines = []
        if DISPLAY_ALL_CONNECTIONS_GRAPHICALLY or HIGHLIGHT_SELECTED_CONNECTIONS:
            for connection in sag.connections:
                source, target = tuple(connection)
                source_left, source_top, _, _ = node_to_corners[source]
                target_left, target_top, _, _ = node_to_corners[target]
                source_x = int(source_left + width_per_box)
                source_y = int(source_top + 0.5 * height_per_box)
                target_x = int(target_left)
                target_y = int(target_top + 0.5 * height_per_box)
                connection_name = self._nodes_to_connection_dash_id(configuration_type, source, target)
                stroke_color = vp.node_type_to_color[sag.name_to_node[source].type_of_tensor]
                if DISPLAY_ALL_CONNECTIONS_GRAPHICALLY:
                    svg_connection_lines.append(dash_svg.Line(
                        id=connection_name,
                        x1=str(source_x), x2=str(target_x), y1=str(source_y), y2=str(target_y),
                        stroke=stroke_color,
                        strokeWidth=1,
                    ))
                if HIGHLIGHT_SELECTED_CONNECTIONS:
                    for n1, n2 in [(source, target), (target, source)]:
                        self.configuration_type_and_node_to_list_of_connections.setdefault((configuration_type, n1),
                                                                                           []).append((
                            source_x, source_y, target_x, target_y, n2,
                            (vp.highlighting_colors[
                                 'highlighted'] if DISPLAY_ALL_CONNECTIONS_GRAPHICALLY else stroke_color)
                        ))
        elements_of_the_graph.append(
            dash_svg.Svg(svg_connection_lines, viewBox=f'0 0 {vp.total_display_width} {vp.total_display_height}'))
        return elements_of_the_graph

    @utilities.runtime_analysis_decorator
    def create_visualization(self):
        app = self.app
        display_type_radio_button_options = ['Tensors', 'Network', 'Notes', 'Graphs']
        if self.external_visualization_file is not None:
            display_type_radio_button_options.append('Visualization')
        # Define the Layout of the App
        app.layout = html.Div(id='main-body-div', style={
            'backgroundColor': 'white',
        }, children=[
            html.Div(id='main-controls-buttons-container', children=[
                dbc.Row([
                    dbc.Col(dcc.RadioItems(id='display-type-radio-buttons', options=display_type_radio_button_options,
                                           value='Tensors', inline=True), width=4),
                    dbc.Col([html.Label("Navigation:"),
                             html.Button(html.I(className="bi bi-arrow-left"), id='navigate-left-button',
                                         className='btn btn-outline-secondary btn-xs', n_clicks=0),
                             html.Button(html.I(className="bi bi-arrow-right"), id='navigate-right-button',
                                         className='btn btn-outline-secondary btn-xs', n_clicks=0),
                             html.Button(html.I(className="bi bi-arrow-up"), id='navigate-up-button',
                                         className='btn btn-outline-secondary btn-xs', n_clicks=0),
                             html.Button(html.I(className="bi bi-arrow-down"), id='navigate-down-button',
                                         className='btn btn-outline-secondary btn-xs', n_clicks=0)
                             ], id='navigation-buttons', width=4),
                    dbc.Col("", width=1),
                    dbc.Col(html.Button('Refresh graphs', id='refresh-kpi-graphs-button', n_clicks=0), width=2),
                    dbc.Col(html.Button('Reload all', id='refresh-button', n_clicks=0), width=1),
                ]),
            ]),
            html.Div(id='dummy-placeholder-output-for-updating-graphs'),  # This dummy stores some data
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
            html.Div(id='controls-selectors-container', children=[
                dbc.Row([
                    dbc.Col(html.Label("Trial"), width=2),
                    dbc.Col(dcc.Dropdown(id='trials-dropdown', options=[], value=None), width=9),
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Role of tensor"), width=2),
                    dbc.Col(dcc.Dropdown(id='role-of-tensor-in-node-dropdown', options=[], value=None), width=9),
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Type of recording"), width=2),
                    dbc.Col(dcc.RadioItems(id='type-of-recording-radio-buttons', options=[], value=None, inline=True),
                            width=9),
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Type of training step"), width=2),
                    dbc.Col(dcc.Dropdown(id='type-of-execution-for-diversity-of-recordings-dropdown', options=[],
                                         value=None), width=9),
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Training step"), width=1),
                    dbc.Col(html.Div([
                        html.Button(html.I(className="bi bi-arrow-left"), id='decrement-training-step-button',
                                    className='btn btn-outline-secondary btn-xs', n_clicks=0),
                        html.Button(html.I(className="bi bi-arrow-right"), id='increment-training-step-button',
                                    className='btn btn-outline-secondary btn-xs', n_clicks=0),
                    ], className='buttons-for-selecting-filters'), width=1),
                    dbc.Col(dcc.Slider(id='training-step-slider', min=0, max=100, step=None, value=None), width=9),
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Iteration"), width=1),
                    dbc.Col(html.Div([
                        html.Button(html.I(className="bi bi-arrow-left"), id='decrement-iteration-button',
                                    className='btn btn-outline-secondary btn-xs', n_clicks=0),
                        html.Button(html.I(className="bi bi-arrow-right"), id='increment-iteration-button',
                                    className='btn btn-outline-secondary btn-xs', n_clicks=0),
                    ], className='buttons-for-selecting-filters'), width=1),
                    dbc.Col(dcc.Slider(id='iteration-slider', min=0, max=0, step=1, value=0), width=9),
                ]),
                dbc.Row([
                    dbc.Col(html.Label("Batch or sample"), width=1),
                    dbc.Col(html.Div([
                        html.Button(html.I(className="bi bi-arrow-left"), id='decrement-batch-index-button',
                                    className='btn btn-outline-secondary btn-xs', n_clicks=0),
                        html.Button(html.I(className="bi bi-arrow-right"), id='increment-batch-index-button',
                                    className='btn btn-outline-secondary btn-xs', n_clicks=0),
                    ], className='buttons-for-selecting-filters'), width=1),
                    dbc.Col(dcc.Dropdown(id='batch-index-dropdown', options=[], value=None), width=9),
                ]),
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
            self.trial_to_kpi_graph_excerpt = {}
            # Load the list of trials
            trials_folder = self.path / 'trials'
            subfolders = [a for a in trials_folder.iterdir() if a.is_dir()]
            options = [{'label': a.name, 'value': a.name} for a in subfolders]
            options.sort(key=lambda a: a['label'])
            return options, options[0]['value']

        @app.callback([Output('dummy-placeholder-output-for-updating-graphs', 'className')],
                      [Input('refresh-kpi-graphs-button', 'n_clicks')],
                      [State('trials-dropdown', 'value')])
        @utilities.runtime_analysis_decorator
        def refresh_kpi_graphs(n_clicks, trials_value):
            self.load_kpi_graphs_for_trial(trials_value)
            return [str(n_clicks)]

        @app.callback(([Output('dummy-for-selecting-a-node', 'className'),
                        Output('type-of-execution-for-diversity-of-recordings-dropdown', 'options'),
                        Output('type-of-execution-for-diversity-of-recordings-dropdown', 'value'),
                        Output('training-step-slider', 'min'),
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
                      ([Input('trials-dropdown', 'value'),
                        Input('type-of-execution-for-diversity-of-recordings-dropdown', 'value'),
                        Input('decrement-training-step-button', 'n_clicks'),
                        Input('increment-training-step-button', 'n_clicks'),
                        Input('decrement-iteration-button', 'n_clicks'),
                        Input('increment-iteration-button', 'n_clicks'),
                        Input('decrement-batch-index-button', 'n_clicks'),
                        Input('increment-batch-index-button', 'n_clicks'),
                        Input('training-step-slider', 'value'),
                        Input('type-of-recording-radio-buttons', 'value'),
                        Input('batch-index-dropdown', 'value'),
                        Input('iteration-slider', 'value'),
                        Input('role-of-tensor-in-node-dropdown', 'value'),
                        Input('dummy-for-selecting-a-node', 'className'),
                        Input('navigate-left-button', 'n_clicks_timestamp'),
                        Input('navigate-right-button', 'n_clicks_timestamp'),
                        Input('navigate-up-button', 'n_clicks_timestamp'),
                        Input('navigate-down-button', 'n_clicks_timestamp')] +
                       [Input(self._node_name_to_dash_id(configuration_type, node), 'n_clicks_timestamp')
                        for configuration_type, sag in self.configuration_type_to_status_and_graph.items()
                        for node in sag.name_to_node.keys()]))
        @utilities.runtime_analysis_decorator
        def update_selectors(
                trials_value, type_of_execution,
                decrement_training_step, increment_training_step,
                decrement_iteration, increment_iteration,
                decrement_batch_index, increment_batch_index,
                training_step_value, type_of_recording_value, batch_index_value,
                iteration_value, role_of_tensor_in_node_value, previous_name_of_selected_node,
                *lsts,
        ):
            navigation_button_clicks = lsts[0:4]
            clicks_per_node = lsts[4:]
            program_is_initializing = (training_step_value is None)

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
            # Selection of valid values for training steps.
            #
            # Training steps and type_of_execution come from file names,
            # while all other attributes are held by the contents of the files that training steps are named after.
            type_of_execution = (type_of_execution or 'any_value')
            recordings_path = self.path / 'trials' / trials_value / 'recordings'
            training_steps_and_types = []
            for possible_path in recordings_path.iterdir():
                match = re.match(r'^([0-9]+)__(.+)$', possible_path.name)
                training_step_value_of_path = int(match.group(1))
                type_of_execution_of_path = match.group(2)
                training_steps_and_types.append((training_step_value_of_path, type_of_execution_of_path))
            list_of_type_of_execution = sorted(list(set(['any_value'] + [a[1] for a in training_steps_and_types])),
                                               key=lambda a: '' if a == 'any_value' else a)
            training_steps = sorted([a[0] for a in training_steps_and_types
                                     if type_of_execution == 'any_value'
                                     or type_of_execution == a[1]])
            assert len(training_steps) > 0
            type_of_execution_options, type_of_execution = create_options_and_value_from_list(
                type_of_execution, list_of_type_of_execution,
                label_maker=lambda a: "Any training step" if a == 'any_value' else a
            )
            training_step_min, training_step_max, training_step_marks, training_step_value = create_slider_data_from_list(
                training_step_value if training_step_value is not None and training_step_value in training_steps else
                training_steps[0],
                training_steps,
            )
            # Increment or decrement the training step if the user clicked the buttons.
            idx = training_steps.index(training_step_value)
            if ctx.triggered_id == 'increment-training-step-button':
                idx = max(0, min(len(training_steps) - 1, idx + 1))
                training_step_value = training_steps[idx]
            elif ctx.triggered_id == 'decrement-training-step-button':
                idx = max(0, min(len(training_steps) - 1, idx - 1))
                training_step_value = training_steps[idx]
            # Get the recordings
            recordings = self.get_recordings_with_caching(trials_value, training_step_value, type_of_execution)
            db: utilities.PseudoDb = recordings.recordings
            if program_is_initializing:
                # Initialize everything
                current_params_dict_for_querying_database = {
                    'record_type': 'data',
                }
                list_of_matches, possible_attribute_values = query_database_using_current_values(
                    [], current_params_dict_for_querying_database,
                )
                assert list_of_matches
                selected_record_values = tuple(list_of_matches[0][0])
            else:
                # Select the node, or change it if the user clicked something.
                name_of_selected_node = self.get_or_change_selected_node(
                    trials_value, type_of_execution,
                    training_step_value, iteration_value, previous_name_of_selected_node,
                    navigation_button_clicks, clicks_per_node,
                )
                original_iteration_value = iteration_value
                #
                # Query the database to determine the best-fitting record to set the current value.
                #
                current_params_dict_for_querying_database = {
                    'training_step': training_step_value,
                    'type_of_tensor_recording': type_of_recording_value,
                    'batch_aggregation': batch_index_value,
                    'iteration': None if self.node_is_a_parameter[name_of_selected_node] else iteration_value,
                    'node_name': name_of_selected_node,
                    'role_within_node': role_of_tensor_in_node_value,
                    'record_type': 'data',
                    'item': None,
                    'metadata': None,
                }
                list_of_matches = []
                possible_attribute_values = {}
                attributes_to_ignore_in_order = ['role_within_node', 'batch_aggregation', 'type_of_tensor_recording',
                                                 'training_step', 'iteration']
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
                attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid = [
                    'iteration', 'batch_aggregation', 'role_within_node']
                for attr in attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid:
                    fallback_list = self.attribute_selection_fallback_values[attr]
                    val = current_params_dict_for_querying_database[attr]
                    value_to_switch_to = next((a for a in fallback_list[::-1] if a in possible_attribute_values[attr]),
                                              None)
                    if value_to_switch_to is not None and val != value_to_switch_to and value_to_switch_to != \
                            fallback_list[-1]:
                        current_params_dict_for_querying_database[attr] = value_to_switch_to
                        list_of_matches, possible_attribute_values = query_database_using_current_values(
                            [], current_params_dict_for_querying_database
                        )
                        assert list_of_matches
                        selected_record_values = tuple(list_of_matches[0][0])
                        for attr_, val_ in zip(db.attributes, selected_record_values):
                            assert attr_ in current_params_dict_for_querying_database, attr_
                            current_params_dict_for_querying_database[attr_] = val_
                    for a in possible_attribute_values[attr]:
                        while a in fallback_list:
                            fallback_list.remove(a)
                for attr in attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid:
                    fallback_list = self.attribute_selection_fallback_values[attr]
                    val = current_params_dict_for_querying_database[attr]
                    fallback_list.append(val)
            #
            # Get the values of the selected record.
            # These overwrite any previously used values and are used from now on.
            #
            training_step_value, type_of_recording_value, batch_index_value, iteration_value, name_of_selected_node, role_of_tensor_in_node_value, record_type, item, metadata = selected_record_values
            assert len(db.attributes) == len(selected_record_values)
            current_params_dict_for_querying_database = {
                k: v for k, v in zip(db.attributes, selected_record_values)
            }
            #
            # Query again, using that record as the filter,
            # to determine which alternative values are legal for each attribute.
            #
            _, possible_attribute_values = query_database_using_current_values(
                [], current_params_dict_for_querying_database
            )
            # Options for the type of recording
            type_of_recording_options, type_of_recording_value = create_options_and_value_from_list(
                type_of_recording_value, possible_attribute_values['type_of_tensor_recording'],
            )
            type_of_recording_options.sort(key=lambda a: a['value'])
            # Options for the batch_index
            batch_index_options, batch_index_value = create_options_and_value_from_list(
                batch_index_value, possible_attribute_values['batch_aggregation'],
                label_maker=lambda a: {
                    'batch_mean': "Mean over the batch",
                    'batch_abs_max': "Maximum absolute value over the batch",
                    'batch_std': "STD over the batch",
                    'has_no_batch_dimension': "Has no batch dimension",
                }.get(a, f"batch index {a}")
            )
            batch_index_options.sort(
                key=lambda a: -1 if a['value'] in ['batch_mean', 'batch_abs_max', 'batch_std'] else (
                    -2 if a['value'] == 'has_no_batch_dimension' else a['value']))
            # Increment or decrement the batch index if the user clicked the buttons.
            idx = [a['value'] for a in batch_index_options].index(batch_index_value)
            if ctx.triggered_id == 'increment-batch-index-button':
                idx = max(0, min(len(batch_index_options) - 1, idx + 1))
                batch_index_value = batch_index_options[idx]['value']
            elif ctx.triggered_id == 'decrement-batch-index-button':
                idx = max(0, min(len(batch_index_options) - 1, idx - 1))
                batch_index_value = batch_index_options[idx]['value']
            # Options for the iteration
            if self.node_is_a_parameter[name_of_selected_node]:
                iteration_options = list(
                    recordings.training_step_to_iteration_to_configuration_type[training_step_value].keys())
                iteration_min, iteration_max, iteration_marks, iteration_value = create_slider_data_from_list(
                    original_iteration_value
                    if self.node_is_a_parameter[name_of_selected_node] and original_iteration_value in iteration_options
                    else iteration_value,
                    iteration_options,
                )
            else:
                iteration_min, iteration_max, iteration_marks, iteration_value = create_slider_data_from_list(
                    iteration_value, possible_attribute_values['iteration'],
                )
            # Increment or decrement the iteration if the user clicked the buttons.
            possible_iteration_values = sorted(list(iteration_marks.keys()))
            idx = possible_iteration_values.index(iteration_value)
            if ctx.triggered_id == 'increment-iteration-button':
                idx = max(0, min(len(possible_iteration_values) - 1, idx + 1))
                iteration_value = possible_iteration_values[idx]
            elif ctx.triggered_id == 'decrement-iteration-button':
                idx = max(0, min(len(possible_iteration_values) - 1, idx - 1))
                iteration_value = possible_iteration_values[idx]
            # Options for the role of the tensor
            role_of_tensor_in_node_options, role_of_tensor_in_node_value = create_options_and_value_from_list(
                role_of_tensor_in_node_value, possible_attribute_values['role_within_node'],
            )
            role_of_tensor_in_node_options.sort(key=lambda a: a['value'])
            #
            # Hide or show different graphs
            #
            configuration_type = recordings.training_step_to_iteration_to_configuration_type[training_step_value][
                iteration_value]
            graph_container_visibilities = [
                'active' if conf_type == configuration_type else 'inactive'
                for conf_type in self.configuration_type_to_status_and_graph.keys()
            ]
            res = [
                      name_of_selected_node,
                      type_of_execution_options, type_of_execution,
                      training_step_min, training_step_max, training_step_marks, training_step_value,
                      type_of_recording_options, type_of_recording_value, batch_index_options, batch_index_value,
                      iteration_min, iteration_max, iteration_marks, iteration_value,
                      role_of_tensor_in_node_options, role_of_tensor_in_node_value,
                  ] + graph_container_visibilities
            return res

        @app.callback(
            [Output('selected-item-details-container', 'children'),
             Output('graph-overlay-for-selections', 'children'),
             Output('main-body-div', 'className')],
            [Input('display-type-radio-buttons', 'value'),
             Input('dummy-for-selecting-a-node', 'className'),
             Input('trials-dropdown', 'value'),
             Input('type-of-execution-for-diversity-of-recordings-dropdown', 'value'),
             Input('training-step-slider', 'value'),
             Input('type-of-recording-radio-buttons', 'value'),
             Input('batch-index-dropdown', 'value'),
             Input('iteration-slider', 'value'),
             Input('role-of-tensor-in-node-dropdown', 'value'),
             Input('dummy-placeholder-output-for-updating-graphs', 'className')]
        )
        @utilities.runtime_analysis_decorator
        def update_kpis_to_display_for_selection(
                display_type_radio_buttons,
                node_name, trials_value, type_of_execution,
                training_step_value, type_of_recording_value,
                batch_index_value, iteration_value, role_of_tensor_in_node_value, _,
        ):
            recordings = self.get_recordings_with_caching(trials_value, training_step_value, type_of_execution)
            configuration_type = recordings.training_step_to_iteration_to_configuration_type[training_step_value][
                iteration_value]
            type_of_execution = recordings.training_step_to_type_of_execution[training_step_value]
            sag = self.configuration_type_to_status_and_graph[configuration_type]
            if self.node_is_a_parameter[node_name]:
                iteration_value = 0
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
            connected_node_names = {a[0] for a in sag.connections if a[1] == node_name} | {a[1] for a in sag.connections
                                                                                           if a[0] == node_name}
            graph_overlay_elements = []
            for node_name_, (left, top, right, bottom) in self.configuration_type_to_node_to_corners[
                configuration_type].items():
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
            if HIGHLIGHT_SELECTED_CONNECTIONS:
                for coordinates in self.configuration_type_and_node_to_list_of_connections.get(
                        (configuration_type, node_name), []):
                    source_x, source_y, target_x, target_y, other_node, stroke_color = coordinates
                    graph_overlay_elements.append(dash_svg.Line(
                        id=f'highlight_connection__{configuration_type}__{node}__{other_node}',
                        x1=str(source_x), x2=str(target_x), y1=str(source_y), y2=str(target_y),
                        stroke=stroke_color,
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
            # Get the data
            filters = {}
            for name, val in [
                ('training_step', training_step_value),
                ('type_of_tensor_recording', type_of_recording_value),
                ('batch_aggregation', batch_index_value),
                ('iteration', iteration_value),
                ('node_name', node_name),
                ('role_within_node', role_of_tensor_in_node_value),
                ('record_type', 'data'),
                ('item', None),
                ('metadata', None),
            ]:
                if val is not None:
                    filters[name] = val
                assert name in ['item', 'metadata'] or val is not None, (name,)
            list_of_matches, possible_attribute_values = db.get_matches(filters)
            assert len(list_of_matches) > 0, (filters,)
            # Get helper information
            filters = {}
            for name, val in [
                ('training_step', training_step_value),
                ('type_of_tensor_recording', 'not_applicable'),
                ('batch_aggregation', 'not_applicable'),
                ('iteration', iteration_value),
                ('node_name', node_name),
                ('role_within_node', role_of_tensor_in_node_value),
                ('record_type', 'meta_information'),
                ('item', None),
                ('metadata', None),
            ]:
                if val is not None:
                    filters[name] = val
                assert name in ['item', 'metadata'] or val is not None, (name,)
            list_of_matches_for_information_things, _ = db.get_matches(filters)
            assert len(list_of_matches_for_information_things) == 2, (len(list_of_matches_for_information_things))
            index_of_item = db.attributes.index('item')
            index_of_metadata = db.attributes.index('metadata')
            tensor_shape = None
            index_of_batch_dimension = None
            for key, val in list_of_matches_for_information_things:
                if key[index_of_item] == 'tensor_shape':
                    tensor_shape = val
                elif key[index_of_item] == 'index_of_batch_dimension':
                    index_of_batch_dimension = val
                else:
                    raise ValueError(key[index_of_item])

            # Create the graphical display

            def optionally_add_tooltip_about_aggregation(item):
                if item != 'mean':
                    return item
                return html.Div([
                    html.Div(
                        [
                            f"{item}",
                            html.Span(
                                "?",
                                id='tooltip-target',
                                style={
                                    'textDecoration': 'underline',
                                    'cursor': 'pointer',
                                    'float': 'right'
                                },
                            ),
                        ]
                    ),
                    dbc.Tooltip(
                        "A note on the order of operations: The aggregation over the batch, if one is selected, "
                        "is applied after the aggregation over the tensor. "
                        "For example, let's say 'Mean over the Batch' is selected, "
                        "and we look at the value of the item 'abs_max'. "
                        "This value is calculated by first calculating the maximum absolute neuron value "
                        "for each tensor in the batch and then taking the mean over those values.",
                        target='tooltip-target',
                    ),
                ])

            rows = []
            for key, val in list_of_matches:
                row = [
                    html.Td(optionally_add_tooltip_about_aggregation(key[index_of_item])),
                    html.Td(key[index_of_metadata]),
                    html.Td(f"", style={'width': '15px', 'background': utilities.number_to_hex(val)} if isinstance(val,
                                                                                                                   numbers.Number) else {}),
                    html.Td(f"{val:17.10f}     -     {' ' if val >= 0.0 else ''}{val:.8e}" if isinstance(val,
                                                                                                         numbers.Number) else val),
                ]
                rows.append(html.Tr(row))
            if display_type_radio_buttons == 'Tensors':
                hide_containers_for_tensors = False
                children = [
                    html.Table(
                        [html.Tr([html.Th(col) for col in
                                  ['Trial', 'Step Type', 'Node', 'Role', 'Tensor Type', 'Tensor Shape', 'Training Step',
                                   'Iteration']])] +
                        [html.Tr([
                            html.Td(val) for val in [
                                trials_value, type_of_execution,
                                node.full_unique_name[len('node__'):],
                                role_of_tensor_in_node_value, node.type_of_tensor,
                                f"[{', '.join([str(a) for a in tensor_shape])}]",
                                training_step_value, iteration_value,
                            ]
                        ])],
                        id='selected-values-summary-table',
                    ),
                    html.Table([html.Tr([html.Th(col) for col in ['Item', '', '', 'value']])] + rows),
                ]
            elif display_type_radio_buttons == 'Network':
                hide_containers_for_tensors = True
                children = [
                    html.Div(f"{self.get_formatted_overview_of_module_parameters(sag)}", className="metadata-div"),
                ]
            elif display_type_radio_buttons == 'Notes':
                hide_containers_for_tensors = True
                children = [
                    html.Div('\n'.join(self.get_notes_for_trial(trials_value)), className="metadata-div"),
                ]
            elif display_type_radio_buttons == 'Graphs':
                hide_containers_for_tensors = True
                children = [
                    html.Div(self.display_kpi_graphs(
                        trials_value, type_of_execution,
                    ), className="kpi-graphs-div"),
                ]
            elif display_type_radio_buttons == 'Visualization':
                hide_containers_for_tensors = False
                children = [
                    html.Div(self.create_external_visualization(
                        recordings, configuration_type, type_of_execution, sag, db,
                        training_step_value, type_of_recording_value, batch_index_value, iteration_value, node_name,
                        role_of_tensor_in_node_value,
                    ), className="external-visualization-div"),
                ]
            else:
                raise ValueError(display_type_radio_buttons)
            class_for_main_div = ('hide-containers-for-tensors' if hide_containers_for_tensors else '')
            return children, graph_overlay_for_selections_children, class_for_main_div

    @utilities.runtime_analysis_decorator
    def get_or_change_selected_node(
            self, trials_value, type_of_execution,
            training_step_value, iteration_value, previous_name_of_selected_node,
            navigation_button_clicks, clicks_per_node,
    ):
        recordings = self.get_recordings_with_caching(trials_value, training_step_value, type_of_execution)
        # Special case: If nothing was clicked, just return the previous_name_of_selected_node
        if max([-1 if a is None else a for a in
                navigation_button_clicks + clicks_per_node]) <= self.last_navigation_click_event_time:
            return previous_name_of_selected_node
        # Identify which Nodes exist in the selected configuration
        names = [
            node
            for _, sag in self.configuration_type_to_status_and_graph.items()
            for node in sag.name_to_node.keys()
        ]
        selected_configuration_type = recordings.training_step_to_iteration_to_configuration_type[training_step_value][
            iteration_value]
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
        if max_val_nodes >= max_val_navigation and max_val_nodes > self.last_navigation_click_event_time:
            self.last_navigation_click_event_time = max_val_nodes
            name_of_selected_node = names[max_index_nodes]
        elif max_val_navigation > self.last_navigation_click_event_time:
            self.last_navigation_click_event_time = max_val_navigation
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
        else:
            name_of_selected_node = previous_name_of_selected_node
        assert name_of_selected_node is not None
        return name_of_selected_node

    @utilities.runtime_analysis_decorator
    def get_formatted_overview_of_module_parameters(self, sag: StatusAndGraph):

        def rec(modules: Dict[str, ModuleRepresentation], indentation_level):
            total_parameters = 0
            lines = []
            for m_name, m_val in modules.items():
                sub_parameter_count = 0
                sub_lines = []
                for p_name, p_val in m_val.parameters.items():
                    numel = math.prod(p_val.shape)
                    sub_parameter_count += numel
                    line = f"{' ' * 4 * (indentation_level + 1)}{numel:18,d} - {p_name} - {p_val.shape}"
                    sub_lines.append(line)
                mod_sub_parameter_count, mod_sub_lines = rec(m_val.submodules, indentation_level + 1)
                sub_parameter_count += mod_sub_parameter_count
                sub_lines.extend(mod_sub_lines)
                head_line = f"{' ' * 4 * indentation_level}{sub_parameter_count:18,d} - {m_name}"
                total_parameters += sub_parameter_count
                lines.append(head_line)
                lines.extend(sub_lines)
            return total_parameters, lines

        total_parameters, lines = rec(sag.modules_and_parameters, 0)
        res = f"Total parameters: {total_parameters:18,d}\n" + "\n".join(lines)
        return res

    @utilities.runtime_analysis_decorator
    def get_notes_for_trial(self, trial_id):
        path = self.path / 'trials' / str(trial_id) / 'notes.json'
        if path.exists():
            with open(path, 'r') as f:
                notes = json.load(f)
            return notes
        return ["No Notes have been recorded for this trial."]

    @utilities.runtime_analysis_decorator
    def display_kpi_graphs(self, trials_value, type_of_execution):
        if trials_value not in self.trial_to_kpi_graph_excerpt:
            self.load_kpi_graphs_for_trial(trials_value)
        graph_excerpt = self.trial_to_kpi_graph_excerpt[trials_value]
        graphs = []
        for type_of_execution, a in graph_excerpt.items():
            for kpi_group, b in a.items():
                plots = []
                for kpi_name, stats in b.items():
                    vals = stats['vals']
                    xs = []
                    ys = []
                    for val in vals:
                        x = val['timepoint']
                        y = val['val']
                        xs.append(x)
                        ys.append(y)
                    plots.append(go.Scatter(x=xs, y=ys, name=kpi_name))
                fig = go.Figure(data=plots, layout=go.Layout(
                    title=f'{type_of_execution}__{kpi_group}',
                    xaxis=dict(title='step'),
                    yaxis=dict(title='value'),
                    legend=dict(x=0, y=1),  # Positioning of the legend (0 to 1, where 0 is left/bottom and 1 is right/top)
                    hovermode='closest'
                ))
                graphs.append(dcc.Graph(figure=fig))
        content_div = html.Div(children=graphs, style={
        })
        return html.Div(children=[
            content_div,
        ], style={
        })

    @utilities.runtime_analysis_decorator
    def load_kpi_graphs_for_trial(self, trials_value):
        if trials_value is None:
            return
        trial_path = self.path / 'trials' / trials_value
        files = [a for a in list(trial_path.iterdir()) if a.stem == 'kpi_graph']
        assert len(files) == 1
        self.trial_to_kpi_graph_excerpt[trials_value] = self.load_file(files[0])

    @utilities.runtime_analysis_decorator
    def create_external_visualization(self, *args):
        try:
            from importlib import util as importlib_util
            spec = importlib_util.spec_from_file_location("module_name", self.external_visualization_file)
            my_module = importlib_util.module_from_spec(spec)
            spec.loader.exec_module(my_module)
            # my_module = importlib.import_module(self.external_visualization_file)
            res = my_module.create_visualization(*args)
            return res
        except Exception as e:
            error_message = utilities.get_error_message_details()
            return f"Failed to execute the file {self.external_visualization_file}:\n{error_message}"
