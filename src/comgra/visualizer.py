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
from typing import Dict, List, Optional, Tuple

import dash
from dash import ctx, dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_svg
import msgpack
import plotly.graph_objs as go

from comgra.objects import ModuleRepresentation, TensorRecordings, TensorReference, TrainingStepConfiguration, \
    SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS, NodeGraphStructure
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
        # 'output': '#00ffff',
        'loss': '#c74fb7',
        'calculated': '#a29fc9',
        'parameter': '#cfc100',
    }
    highlighting_colors = {
        'selected': '#ff1700',
        'highlighted': '#9a0000',
    }


vp = VisualizationParameters()


class CustomDash(dash.Dash):
    def interpolate_index(self, **kwargs):
        extra_script = """
        <script>
            window.addEventListener("keydown", (event) => {
                console.log(event);
                if (event.key == 'ArrowUp') {
                    document.getElementById("navigate-up-button").click();
                    event.preventDefault();
                }
                if (event.key == 'ArrowRight') {
                    document.getElementById("navigate-right-button").click();
                    event.preventDefault();
                }
                if (event.key == 'ArrowDown') {
                    document.getElementById("navigate-down-button").click();
                    event.preventDefault();
                }
                if (event.key == 'ArrowLeft') {
                    document.getElementById("navigate-left-button").click();
                    event.preventDefault();
                }
            });
        </script>
        """
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
            {extra_script}
            {renderer}
        </footer>
    </body>
</html>
        '''.format(**kwargs, extra_script=extra_script)


class Visualization:
    """
    Keeps track of KPIs over the course of the Experiment and creates visualizations from them.
    """

    def __init__(self, path, debug_mode, external_visualization_file, restart_signal_queue):
        super().__init__()
        utilities.DEBUG_MODE = debug_mode
        self.debug_mode = debug_mode
        self.path: Path = path
        self.external_visualization_file = external_visualization_file
        self.restart_signal_queue = restart_signal_queue
        assert path.exists(), path
        assets_path = Path(__file__).absolute().parent.parent / 'assets'
        assert assets_path.exists(), "If this fails, files have been moved."
        self.app = CustomDash(__name__, assets_folder=str(assets_path),
                              external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.BOOTSTRAP])
        self.ngs_hash_to_ngs: Dict[str, NodeGraphStructure] = {}
        self.ngs_hash_to_node_to_corners: Dict[str, Dict[str, Tuple[int, int, int, int]]] = {}
        self.ngs_hash_to_grid_of_nodes: Dict[str, List[List[str]]] = {}
        self.ngs_hash_to_node_to_list_of_connections: Dict[
            str, Dict[str, List[Tuple[int, int, int, int, str, str]]]] = {}
        self.ngs_hash_to_graph_container_dash_id: Dict[str, str] = {}
        self.ngs_hash_to_node_to_dash_id: Dict[str, Dict[str, str]] = {}
        self.list_of_unique_node_dash_ids: List[str] = []
        self._sanity_check_cache_to_avoid_duplicates = {}
        self.cache_for_training_step_configuration_and_recordings = {}
        self.attribute_selection_fallback_values = collections.defaultdict(list)
        self.last_navigation_click_event_time = -1
        self.trial_to_kpi_graph_excerpt = {}
        self.last_selected_record_values: Optional[Tuple] = None

    def _nodes_to_connection_dash_id(self, ngs_hash, source, target):
        return f"connection__{self.ngs_hash_to_node_to_dash_id[ngs_hash][source]}__" \
               f"{self.ngs_hash_to_node_to_dash_id[ngs_hash][target]}"

    def run_server(self, port, use_reloader):
        #
        # Load data that can only be loaded once, because Divs depend on it.
        #
        counter_for_container_id = 0
        counter_for_node_id = 0
        assert not self.ngs_hash_to_graph_container_dash_id
        assert not self.ngs_hash_to_node_to_dash_id
        # For each unique graph structure...
        for ngs_file in (self.path / 'node_graph_structure').iterdir():
            with open(ngs_file, 'rb') as f:
                ngs: NodeGraphStructure = pickle.load(f)
            assert ngs.node_graph_hash == ngs_file.stem, (ngs.node_graph_hash, ngs_file.name)
            self.ngs_hash_to_ngs[ngs.node_graph_hash] = ngs
            # Create a dash ID for the graph container
            graph_container_dash_id = f'graph_container__{counter_for_container_id}'
            counter_for_container_id += 1
            self.ngs_hash_to_graph_container_dash_id[ngs.node_graph_hash] = graph_container_dash_id
            # Create a dash ID for each node of the graph
            node_to_dash_id = {}
            for node in ngs.name_to_node.keys():
                node_dash_id = node.replace('.', '__')
                node_dash_id = f"confnode__{graph_container_dash_id}__{node_dash_id}__{counter_for_node_id}"
                assert (node_dash_id not in self._sanity_check_cache_to_avoid_duplicates or
                        self._sanity_check_cache_to_avoid_duplicates[node_dash_id] == node), \
                    f"Programming error: Two nodes have the same representation: " \
                    f"{node}, {self._sanity_check_cache_to_avoid_duplicates[node_dash_id]}, {node_dash_id}"
                self._sanity_check_cache_to_avoid_duplicates[node_dash_id] = node
                counter_for_node_id += 1
                node_to_dash_id[node] = node_dash_id
            self.ngs_hash_to_node_to_dash_id[ngs.node_graph_hash] = node_to_dash_id
        assert not self.list_of_unique_node_dash_ids
        for node_to_dash_id in self.ngs_hash_to_node_to_dash_id.values():
            for node_dash_id in node_to_dash_id.values():
                if node_dash_id not in self.list_of_unique_node_dash_ids:
                    self.list_of_unique_node_dash_ids.append(node_dash_id)
        print(
            f"There are {len(self.ngs_hash_to_node_to_dash_id)} different graph layouts "
            f"to visualize, with a total of {len(self.list_of_unique_node_dash_ids)} unique nodes."
        )
        #
        # Visualize
        #
        self.create_visualization()
        self.app.run_server(debug=self.debug_mode, port=port, use_reloader=use_reloader)

    @utilities.runtime_analysis_decorator
    def get_training_step_configuration_and_recordings(
            self, trials_value, training_step_value, type_of_execution
    ) -> Tuple[TrainingStepConfiguration, TensorRecordings]:
        with LOCK_FOR_RECORDINGS:
            key = (trials_value, training_step_value,)
            res = self.cache_for_training_step_configuration_and_recordings.get(key, None)
            if res is None:
                # Load the configuration for the training step
                # Note / reminder: Do I need to filter by type_of_execution?
                # The training_step determines the type_of_execution so there should actually be no ambiguity here.
                # However, other parts of the code use the type_of_execution in the names of these files
                # to determine the list of valid training_steps for a given selected value of type_of_execution
                # without having to load those training_steps first.
                recordings_path_base = self.path / 'trials' / trials_value / 'configurations'
                training_step_configuration_path = recordings_path_base / f'{training_step_value}.pkl'
                with open(training_step_configuration_path, 'rb') as f:
                    tsc: TrainingStepConfiguration = pickle.load(f)
                assert type_of_execution == 'any_value' or tsc.type_of_execution == type_of_execution, \
                    (tsc.type_of_execution, type_of_execution)
                # Load the recordings of the training step
                recordings_path_base = self.path / 'trials' / trials_value / 'recordings'
                if type_of_execution == 'any_value':
                    for recordings_path in recordings_path_base.iterdir():
                        if recordings_path.name.startswith(f'{training_step_value}__'):
                            break
                else:
                    recordings_path = recordings_path_base / f'{training_step_value}__{type_of_execution}'
                recording_files = sorted(list(recordings_path.iterdir()))
                recordings = None
                for recording_file in recording_files:
                    new_recordings_data = self.load_file(recording_file)
                    new_recordings: TensorRecordings = TensorRecordings(**new_recordings_data)
                    new_recordings.recordings = utilities.PseudoDb([]).deserialize(new_recordings.recordings)
                    if recordings is None:
                        recordings = new_recordings
                    else:
                        recordings.update_with_more_recordings(new_recordings)
                recordings.recordings.create_index(
                    ['training_step', 'record_type', 'node_name', 'role_within_node', 'batch_aggregation', 'iteration'],
                    # Special case while accessing the database:
                    # We never want tensors from iteration -1, since those are dummy values
                    # If there is no filter on the iteration, then add one that excludes -1
                    filter_values_to_ignore={
                        'iteration': {-1},
                    }
                )
                # Save both
                res = (tsc, recordings)
                self.cache_for_training_step_configuration_and_recordings[key] = res
            return res

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
            self, ngs_hash: str,
    ) -> List:
        assert ngs_hash not in self.ngs_hash_to_node_to_corners, ngs_hash
        node_to_corners = self.ngs_hash_to_node_to_corners.setdefault(ngs_hash, {})
        grid_of_nodes = self.ngs_hash_to_grid_of_nodes.setdefault(ngs_hash, [])
        ngs = self.ngs_hash_to_ngs[ngs_hash]
        highest_number_of_nodes = max(len(a) for a in ngs.dag_format)
        height_per_box = int((vp.total_display_height - vp.padding_top - vp.padding_bottom) / (
                highest_number_of_nodes + (
                highest_number_of_nodes - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        width_per_box = int((vp.total_display_width - vp.padding_left - vp.padding_right) / (
                len(ngs.dag_format) + (len(ngs.dag_format) - 1) * vp.ratio_of_space_between_nodes_to_node_size))
        nodes_that_have_dependencies = {
            source_and_target[1] for source_and_target in ngs.node_connections
        }
        nodes_that_have_dependents = {
            source_and_target[0] for source_and_target in ngs.node_connections
        }
        elements_of_the_graph = []
        for i, nodes in enumerate(ngs.dag_format):
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
                # Note: This formula was determined experimentally to be "good enough".
                # Low-priority task: Replace it with a better, CSS-based solution.
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
                node_type = ngs.name_to_node[node].type_of_tensor
                text_in_node = self.clean_name_of_tensor_or_node('tensor', node_type, node[len(common_prefix):])
                text_on_mouseover = self.clean_name_of_tensor_or_node('node', node_type, node)
                appropriate_font_size_for_text_in_node = get_appropriate_font_size_for_text_in_node(width_per_box,
                                                                                                    text_in_node)
                node_classes = 'node'
                if node not in nodes_that_have_dependencies and node_type != 'parameter':
                    node_classes += ' node-without-dependency'
                if node not in nodes_that_have_dependents and node_type != 'parameter':
                    node_classes += ' node-without-dependent'
                elements_of_the_graph.append(
                    html.Div(id=self.ngs_hash_to_node_to_dash_id[ngs_hash][node], className=node_classes, style={
                        'width': f'{width_per_box}px',
                        'height': f'{height_per_box}px',
                        'left': f'{left}px',
                        'top': f'{top}px',
                        'background': vp.node_type_to_color[node_type],
                    }, title=text_on_mouseover, children=(
                        [html.Div(f'{text_in_node}', className='node-name', style={
                            'font-size': f'{appropriate_font_size_for_text_in_node}px'
                        })] if DISPLAY_NAMES_ON_NODES_GRAPHICALLY else [])
                             ))
        svg_connection_lines = []
        if DISPLAY_ALL_CONNECTIONS_GRAPHICALLY or HIGHLIGHT_SELECTED_CONNECTIONS:
            for connection in ngs.node_connections:
                source, target = tuple(connection)
                source_left, source_top, _, _ = node_to_corners[source]
                target_left, target_top, _, _ = node_to_corners[target]
                source_x = int(source_left + width_per_box)
                source_y = int(source_top + 0.5 * height_per_box)
                target_x = int(target_left)
                target_y = int(target_top + 0.5 * height_per_box)
                connection_name = self._nodes_to_connection_dash_id(ngs_hash, source, target)
                stroke_color_by_source = vp.node_type_to_color[ngs.name_to_node[source].type_of_tensor]
                stroke_color_by_target = vp.node_type_to_color[ngs.name_to_node[target].type_of_tensor]
                if DISPLAY_ALL_CONNECTIONS_GRAPHICALLY:
                    svg_connection_lines.append(dash_svg.Line(
                        id=connection_name,
                        x1=str(source_x), x2=str(target_x), y1=str(source_y), y2=str(target_y),
                        stroke=stroke_color_by_source,
                        strokeWidth=1,
                    ))
                if HIGHLIGHT_SELECTED_CONNECTIONS:
                    for n1, n2, color_by_other_node in [
                        (source, target, stroke_color_by_target),
                        (target, source, stroke_color_by_source),
                    ]:
                        self.ngs_hash_to_node_to_list_of_connections.setdefault(
                            ngs_hash, {}).setdefault(n1, []).append((
                            source_x, source_y, target_x, target_y, n2,
                            (vp.highlighting_colors[
                                 'highlighted'] if DISPLAY_ALL_CONNECTIONS_GRAPHICALLY else color_by_other_node)
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
                    dbc.Col(html.Button('Refresh graphs', id='refresh-kpi-graphs-button', n_clicks=0), width=1),
                    dbc.Col(html.Button('Reload data', id='refresh-button', n_clicks=0), width=1),
                    dbc.Col(html.Button('Restart server', id='restart-button', n_clicks=0), width=1),
                ]),
            ]),
            html.Div(id='dummy-placeholder-output-for-updating-graphs'),  # This dummy stores some data
            html.Div(id='dummy-for-selecting-a-node'),  # This dummy stores some data
            html.Div(id='dummy-placeholder-output-for-restarting-1'),  # This dummy is necessary for Dash not to complain
            html.Div(id='dummy-placeholder-output-for-restarting-2'),  # This dummy is necessary for Dash not to complain
            html.Div(id='controls-selectors-container', children=[
                dbc.Row([
                    dbc.Col(html.Label("Trial"), width=2),
                    dbc.Col(dcc.Dropdown(id='trials-dropdown', options=[], value=None), width=9),
                ], className="display-even-when-kpi-graphs-are-selected"),
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
                dbc.Row([
                    dbc.Col(html.Label("Role of tensor"), width=2),
                    dbc.Col(dcc.Dropdown(id='role-of-tensor-in-node-dropdown', options=[], value=None), width=9),
                ]),
            ]),
            html.Div(id='graph-container', style={
                'position': 'relative',
                'width': f'{vp.total_display_width}px',
                'height': f'{vp.total_display_height}px',
                'border': '1px solid black',
            }, children=[
                            html.Div(id=graph_container_dash_id, style={
                                'position': 'relative',
                                'width': f'{vp.total_display_width}px',
                                'height': f'{vp.total_display_height}px',
                            }, children=self.create_nodes_and_arrows(ngs_hash)
                                     )
                            for ngs_hash, graph_container_dash_id in self.ngs_hash_to_graph_container_dash_id.items()
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
            html.Div(id='selected-item-details-container', children=[
            ]),
        ])

        @app.callback([Output('trials-dropdown', 'options'),
                       Output('trials-dropdown', 'value'),
                       Output('restart-button', 'n_clicks')],
                      [Input('refresh-button', 'n_clicks')])
        @utilities.runtime_analysis_decorator
        def refresh_all_data(n_clicks):
            # Reset caches
            self.cache_for_training_step_configuration_and_recordings = {}
            self.trial_to_kpi_graph_excerpt = {}
            # If any graphs changed, a complete refresh is required, so trigger the restart-button
            # Do NOT do this in debug mode, because that interferes with the restart ability of Dash
            # because it also enables use_reloader.
            server_restart_required = False
            for ngs_file in (self.path / 'node_graph_structure').iterdir():
                node_graph_hash = ngs_file.stem
                if node_graph_hash not in self.ngs_hash_to_ngs:
                    if not self.debug_mode:
                        server_restart_required = True
            # Load the list of trials
            trials_folder = self.path / 'trials'
            subfolders = [a for a in trials_folder.iterdir() if a.is_dir()]
            options = [{'label': a.name, 'value': a.name} for a in subfolders]
            options.sort(key=lambda a: a['label'])
            return options, options[0]['value'], (1 if server_restart_required else 0)

        app.clientside_callback(
            """
            function(n_clicks) {
                if (n_clicks > 0) {
                    window.location.reload();
                }
                return 'ignore_this';
            }
            """,
            Output('dummy-placeholder-output-for-restarting-2', 'className'),
            Input('restart-button', 'n_clicks')
        )

        @app.callback([Output('dummy-placeholder-output-for-restarting-1', 'className')],
                      [Input('restart-button', 'n_clicks')])
        @utilities.runtime_analysis_decorator
        def reload_server(n_clicks):
            if n_clicks > 0:
                assert not self.debug_mode, ("If debug_mode is enabled, use_reloader is also enabled "
                                             "(as of the time of this writing), which breaks multiprocessing.")
                self.restart_signal_queue.put('restart')
            return ['ignore_this']

        @app.callback([Output('dummy-placeholder-output-for-updating-graphs', 'className')],
                      [Input('refresh-kpi-graphs-button', 'n_clicks')],
                      [State('trials-dropdown', 'value')])
        @utilities.runtime_analysis_decorator
        def refresh_kpi_graphs(n_clicks, trials_value):
            self.load_kpi_graphs_for_trial(trials_value)
            return [str(n_clicks)]

        @app.callback([Output('dummy-for-selecting-a-node', 'className'),
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
                      [Output(graph_container_dash_id, 'className')
                       for graph_container_dash_id
                       in self.ngs_hash_to_graph_container_dash_id.values()
                       ],
                      [Input('decrement-training-step-button', 'n_clicks'),
                       Input('increment-training-step-button', 'n_clicks'),
                       Input('decrement-batch-index-button', 'n_clicks'),
                       Input('increment-batch-index-button', 'n_clicks'),
                       Input('decrement-iteration-button', 'n_clicks'),
                       Input('increment-iteration-button', 'n_clicks'),
                       Input('training-step-slider', 'value'),
                       Input('training-step-slider', 'marks'),
                       Input('batch-index-dropdown', 'value'),
                       Input('batch-index-dropdown', 'options'),
                       Input('iteration-slider', 'value'),
                       Input('iteration-slider', 'marks'),
                       Input('trials-dropdown', 'value'),
                       Input('type-of-execution-for-diversity-of-recordings-dropdown', 'value'),
                       Input('type-of-recording-radio-buttons', 'value'),
                       Input('role-of-tensor-in-node-dropdown', 'value'),
                       Input('dummy-for-selecting-a-node', 'className'),
                       Input('navigate-left-button', 'n_clicks_timestamp'),
                       Input('navigate-right-button', 'n_clicks_timestamp'),
                       Input('navigate-up-button', 'n_clicks_timestamp'),
                       Input('navigate-down-button', 'n_clicks_timestamp')] +
                      [Input(node_dash_id, 'n_clicks_timestamp')
                       for node_dash_id in self.list_of_unique_node_dash_ids])
        @utilities.runtime_analysis_decorator
        def update_selectors(
                decrement_training_step, increment_training_step,
                decrement_batch_index, increment_batch_index,
                decrement_iteration, increment_iteration,
                training_step_value, training_step_marks,
                batch_aggregation_value, batch_aggregation_options,
                iteration_value, iteration_marks,
                trials_value, type_of_execution,
                type_of_recording_value,
                role_of_tensor_in_node_value, previous_name_of_selected_node,
                *lsts,
        ):
            navigation_button_clicks = lsts[0:4]
            clicks_per_node = lsts[4:]
            program_is_initializing = (training_step_value is None)
            #
            # Preliminaries: The arrow buttons that increment / decrement values
            #
            if not program_is_initializing:
                # Increment or decrement the training step if the user clicked the buttons.
                assert all(k == v for k, v in training_step_marks.items())
                possible_training_step_values = sorted([int(k) for k in training_step_marks.keys()])
                idx = possible_training_step_values.index(training_step_value)
                if ctx.triggered_id == 'increment-training-step-button':
                    idx = max(0, min(len(possible_training_step_values) - 1, idx + 1))
                    training_step_value = possible_training_step_values[idx]
                elif ctx.triggered_id == 'decrement-training-step-button':
                    idx = max(0, min(len(possible_training_step_values) - 1, idx - 1))
                    training_step_value = possible_training_step_values[idx]
                # Increment or decrement the sample index if the user clicked the buttons.
                idx = [a['value'] for a in batch_aggregation_options].index(batch_aggregation_value)
                if ctx.triggered_id == 'increment-batch-index-button':
                    idx = max(0, min(len(batch_aggregation_options) - 1, idx + 1))
                    batch_aggregation_value = batch_aggregation_options[idx]['value']
                elif ctx.triggered_id == 'decrement-batch-index-button':
                    idx = max(0, min(len(batch_aggregation_options) - 1, idx - 1))
                    batch_aggregation_value = batch_aggregation_options[idx]['value']
                # Increment or decrement the iteration if the user clicked the buttons.
                assert all(k == v for k, v in iteration_marks.items())
                possible_iteration_values = sorted([int(k) for k in iteration_marks.keys()])
                idx = possible_iteration_values.index(iteration_value)
                if ctx.triggered_id == 'increment-iteration-button':
                    idx = max(0, min(len(possible_iteration_values) - 1, idx + 1))
                    iteration_value = possible_iteration_values[idx]
                elif ctx.triggered_id == 'decrement-iteration-button':
                    idx = max(0, min(len(possible_iteration_values) - 1, idx - 1))
                    iteration_value = possible_iteration_values[idx]
            
            #
            # Start working on actual selectors
            #

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

            def query_database_using_current_values(
                    attributes_to_ignore, current_params_dict_for_querying_database,
                    get_first_match_only_and_recalculate_possible_attribute_values=False
            ):
                filters = {}
                for name, val in current_params_dict_for_querying_database.items():
                    if val is not None and name not in attributes_to_ignore:
                        filters[name] = val
                list_of_matches, possible_attribute_values = db.get_matches(filters)
                if get_first_match_only_and_recalculate_possible_attribute_values and list_of_matches:
                    selected_record_values = list_of_matches[0][0]
                    for attr_, val_ in zip(db.attributes, selected_record_values):
                        assert attr_ in current_params_dict_for_querying_database, attr_
                        current_params_dict_for_querying_database[attr_] = val_
                    # Rerun this, because possible_attribute_values is now a subset of before,
                    # since we are limiting the search to selected_record_values
                    list_of_matches, possible_attribute_values = query_database_using_current_values(
                        [], current_params_dict_for_querying_database
                    )
                    assert selected_record_values == tuple(list_of_matches[0][0]), "This shouldn't have changed."
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
            # Get the recordings
            tsc, recordings = self.get_training_step_configuration_and_recordings(trials_value, training_step_value, type_of_execution)
            db: utilities.PseudoDb = recordings.recordings
            if program_is_initializing:
                # Initialize everything
                current_params_dict_for_querying_database = {
                    'record_type': 'data',
                    # Otherwise it might select something with iteration==-1,
                    # which is a useful dummy but invalid to display
                    'iteration': 0,
                }
                list_of_matches, _ = query_database_using_current_values(
                    [], current_params_dict_for_querying_database,
                    get_first_match_only_and_recalculate_possible_attribute_values=False,
                )
                assert list_of_matches
                selected_record_values = tuple(list_of_matches[0][0])
            else:
                # Select the node, or change it if the user clicked something.
                name_of_selected_node, node_was_explicitly_selected = self.get_or_change_selected_node(
                    trials_value, type_of_execution,
                    training_step_value, iteration_value, previous_name_of_selected_node,
                    navigation_button_clicks, clicks_per_node,
                )
                #
                # Try to find a perfect match first,
                # but if it is not possible to find a match, set the filters to None one by one until a match is found.
                # This can happen e.g. if you select a different node
                # and there is no role_within_node for which that is valid
                #
                # Notes on attribute dependencies for future reference if I decide to refactor this:
                # type_of_execution -> training_step - but this is already fully covered by the code above
                # training_step -> iteration
                # iteration -> node
                # node -> role
                # node -> batch_aggregation
                # type_of_recording_value -> [] - can be ignored, all items should have the same recording types: forward, gradients, etc.
                # other fields should not need to be relaxed, but may be included because why not
                #
                # Goals:
                # * type_of_tensor_recording changed --> ignore training_step IFF necessary (it might still be valid)
                #     this is already ensured earlier in this function. training_step_value is valid.
                # * training step changed, num_iterations changes --> IFF a node is selected that only exists in some iterations, such as the loss, then change the iteration to match.
                # * when keeping node --> keep role & batch_aggregation
                # * explore nodes, then switch back to node after a while --> reset role & batch_aggregation
                #
                # The only way the iteration can change automatically is if (1) it wasn't manually changed
                # and (2) it is impossible to find a match with the iteration.
                # The only way it can be impossible for the iteration to match is if if the training step changed
                # because now that iteration might have a different graph and the combination iteration/node is invalid
                # In that scenario, we would prefer to keep the node selected and change the iteration.
                # This happens by relaxing the requirement on the iteration, so that the node can be selected,
                # followed by an attempt to restore the iteration using the fallback lists, just in case.
                # Meanwhile, the role_within_node and batch_aggregation both depend only on the node and are
                # of secondary importance.
                # We relax the requirements on them first, but try to restore them with fallback lists.
                # End result:
                # When you change the training_step, it will try to adjust the iteration to keep the node selected.
                # This is useful if you e.g. have the loss selected,
                # which is on the final iteration of each training_step.
                # When you switch between nodes, switching back to a previously visited one will restore the role
                # you had selected on that node.
                # When you switch iteration, and it changes nodes, it will NOT switch back to the previously selected
                # node when you switch back to this iteration.
                # This is on purpose, because it seemed like a forceful switch would more often be annoying than
                # intended.
                # It is also a technical issue: If I do decide to switch back to nodes, the fallback mechanic needs
                # to be improved: 'role_within_node' and 'batch_aggregation' both depend on 'node_name',
                # so if I change later with fallback then I have to first remove the other two from the filters,
                # in case the node does not have those values, and then afterward add them again, in case the node
                # does have those values, and we don't want to change them by accident.
                #
                current_params_dict_for_querying_database = {
                    'training_step': training_step_value,
                    'type_of_tensor_recording': type_of_recording_value,
                    'batch_aggregation': batch_aggregation_value,
                    'iteration': iteration_value,
                    'node_name': name_of_selected_node,
                    'role_within_node': role_of_tensor_in_node_value,
                    'record_type': 'data',
                    'item': None,
                    'metadata': None,
                }
                # Determine which attribute was changed explicitly
                relevant_attributes_that_were_changed_since_the_last_call_of_this_function = set()
                for attr in ['training_step', 'type_of_tensor_recording', 'batch_aggregation',
                             'iteration', 'node_name', 'role_within_node']:
                    idx = db.attributes.index(attr)
                    if current_params_dict_for_querying_database[attr] != self.last_selected_record_values[idx]:
                        relevant_attributes_that_were_changed_since_the_last_call_of_this_function.add(attr)
                # Iteratively relax filters until we have a match
                list_of_matches = []
                possible_attribute_values = {}
                attributes_that_need_special_handling_with_relaxing_requirements_and_fallback_values = [
                    'role_within_node', 'batch_aggregation', 'iteration', 'node_name',
                ]
                attributes_to_relax_in_order_until_a_match_is_found = [
                    # Do not relax filters on attributes that have just been explicitly selected by the user
                    a for a in attributes_that_need_special_handling_with_relaxing_requirements_and_fallback_values
                    if a not in relevant_attributes_that_were_changed_since_the_last_call_of_this_function
                ]
                for i in range(len(attributes_to_relax_in_order_until_a_match_is_found) + 1):
                    list_of_matches, possible_attribute_values = query_database_using_current_values(
                        attributes_to_relax_in_order_until_a_match_is_found[:i], current_params_dict_for_querying_database,
                        get_first_match_only_and_recalculate_possible_attribute_values=True,
                    )
                    if list_of_matches:
                        break
                assert list_of_matches
                selected_record_values = tuple(list_of_matches[0][0])
                for attr, val in zip(db.attributes, selected_record_values):
                    assert attr in current_params_dict_for_querying_database, attr
                    current_params_dict_for_querying_database[attr] = val
                # Reactivate older values for attributes from the fallback lists.
                # Switch to the last possible value in the fallback list,
                # unless that attribute was explicitly changed by the user.
                # Then update the fallback lists for all attributes by removing all alternative values,
                # and add the new selected value to the end of it.
                attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid = \
                    attributes_to_relax_in_order_until_a_match_is_found
                for attr in attributes_to_consider_for_falling_back_to_previous_selections_that_were_temporarily_invalid:
                    fallback_list = self.attribute_selection_fallback_values[attr]
                    val = current_params_dict_for_querying_database[attr]
                    value_to_switch_to = next(
                        (a for a in fallback_list[::-1] if a in possible_attribute_values[attr]), None
                    )
                    if value_to_switch_to is not None and val != value_to_switch_to:
                        current_params_dict_for_querying_database[attr] = value_to_switch_to
                        list_of_matches, _ = query_database_using_current_values(
                            [], current_params_dict_for_querying_database,
                            get_first_match_only_and_recalculate_possible_attribute_values=True,
                        )
                        assert len(list_of_matches) == 1
                        selected_record_values = tuple(list_of_matches[0][0])
                        for attr_, val_ in zip(db.attributes, selected_record_values):
                            assert attr_ in current_params_dict_for_querying_database, attr_
                            current_params_dict_for_querying_database[attr_] = val_
                    for a in possible_attribute_values[attr]:
                        while a in fallback_list:
                            fallback_list.remove(a)
                # Update all fallback lists, including for the attributes that were changed
                for attr in attributes_that_need_special_handling_with_relaxing_requirements_and_fallback_values:
                    fallback_list = self.attribute_selection_fallback_values[attr]
                    val = current_params_dict_for_querying_database[attr]
                    fallback_list.append(val)
            #
            # Get the values of the selected record.
            # These overwrite any previously used values and are used from now on.
            #
            (training_step_value, type_of_recording_value, batch_aggregation_value, iteration_value, name_of_selected_node,
             role_of_tensor_in_node_value, record_type, item, metadata) = selected_record_values
            assert len(db.attributes) == len(selected_record_values)
            current_params_dict_for_querying_database = {
                k: v for k, v in zip(db.attributes, selected_record_values)
            }
            # Save the selected_record_values
            self.last_selected_record_values = selected_record_values
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
            batch_aggregation_options, batch_aggregation_value = create_options_and_value_from_list(
                batch_aggregation_value, possible_attribute_values['batch_aggregation'],
                label_maker=lambda a: {
                    'batch_mean': "Mean over the batch",
                    'batch_abs_max': "Maximum absolute value over the batch",
                    'batch_std': "STD over the batch",
                    'has_no_batch_dimension': "Has no batch dimension",
                }.get(a, a)
            )
            def extract_sorting_key_from_sample_string(sample_string):
                m = re.match(r'Sample (\d)+(.*)', sample_string)
                return m[2], int(m[1])
            order_of_batch_aggregation_options = {
                'batch_mean': -5,
                'batch_abs_max': -4,
                'batch_std': -3,
                'has_no_batch_dimension': -2,
            }
            batch_aggregation_options.sort(
                key=lambda a: ('', order_of_batch_aggregation_options[a['value']]) if a['value'] in order_of_batch_aggregation_options
                else extract_sorting_key_from_sample_string(a['value'])
            )
            # Options for the iteration
            always_show_all_iterations = True
            if always_show_all_iterations:
                # Note: These values are only retrieved here and not at the start of this function
                # because some arguments may be unset during initialization
                tsc, recordings = self.get_training_step_configuration_and_recordings(trials_value, training_step_value, type_of_execution)
                iteration_min, iteration_max, iteration_marks, iteration_value = create_slider_data_from_list(
                    iteration_value, list(range(len(tsc.graph_configuration_per_iteration))),
                )
            else:
                iteration_min, iteration_max, iteration_marks, iteration_value = create_slider_data_from_list(
                    iteration_value, possible_attribute_values['iteration'],
                )
            # Options for the role of the tensor
            role_of_tensor_in_node_options, role_of_tensor_in_node_value = create_options_and_value_from_list(
                role_of_tensor_in_node_value, possible_attribute_values['role_within_node'],
            )
            role_of_tensor_in_node_options.sort(key=lambda a: a['value'])
            #
            # Hide or show different graphs
            #
            ngs_hash = tsc.graph_configuration_per_iteration[iteration_value].hash_of_node_graph_structure
            graph_container_visibilities = [
                'active' if k == ngs_hash else 'inactive'
                for k in self.ngs_hash_to_graph_container_dash_id.keys()
            ]
            assert len(graph_container_visibilities) == len(self.ngs_hash_to_graph_container_dash_id)
            assert len([a for a in graph_container_visibilities if a == 'active']) == 1, \
                (ngs_hash, len([a for a in graph_container_visibilities if a == 'active']))
            # Update the self.last_selected_record_values again, because they may have changed compared to
            # earlier in this function
            tmp = (training_step_value, type_of_recording_value, batch_aggregation_value, iteration_value,
                   name_of_selected_node, role_of_tensor_in_node_value, record_type, item, metadata)
            assert len(tmp) == len(self.last_selected_record_values)
            self.last_selected_record_values = tmp
            res = [
                      name_of_selected_node,
                      type_of_execution_options, type_of_execution,
                      training_step_min, training_step_max, training_step_marks, training_step_value,
                      type_of_recording_options, type_of_recording_value, batch_aggregation_options, batch_aggregation_value,
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
                batch_aggregation_value, iteration_value, role_of_tensor_in_node_value, _,
        ):
            tsc, recordings = self.get_training_step_configuration_and_recordings(trials_value, training_step_value, type_of_execution)
            graph_config = tsc.graph_configuration_per_iteration[iteration_value]
            ngs_hash = graph_config.hash_of_node_graph_structure
            ngs = self.ngs_hash_to_ngs[ngs_hash]
            tgs = graph_config.tensor_graph_structure
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
            node = ngs.name_to_node[node_name]
            connected_node_names = {
                                       a[0] for a in ngs.node_connections if a[1] == node_name
                                   } | {
                                       a[1] for a in ngs.node_connections if a[0] == node_name
                                   }

            def check_if_connection_is_active_for_this_role_within_node(node0, node1):
                return any(
                    (dependency_and_dependent[0].role_within_node == role_of_tensor_in_node_value) and
                    (dependency_and_dependent[0].node_name == node0 and dependency_and_dependent[1].node_name == node1)
                    for dependency_and_dependent in tgs.tensor_connections
                ) or any(
                    (dependency_and_dependent[1].role_within_node == role_of_tensor_in_node_value) and
                    (dependency_and_dependent[1].node_name == node0 and dependency_and_dependent[0].node_name == node1)
                    for dependency_and_dependent in tgs.tensor_connections
                )
            # Create the elements
            graph_overlay_elements = []
            for node_name_, (left, top, right, bottom) in self.ngs_hash_to_node_to_corners[ngs_hash].items():
                if node_name_ == node_name:
                    color = vp.highlighting_colors['selected']
                    stroke_width = 3
                elif node_name_ in connected_node_names:
                    color = vp.highlighting_colors['highlighted']
                    connection_is_active = check_if_connection_is_active_for_this_role_within_node(node_name, node_name_)
                    stroke_width = 3 if connection_is_active else 1
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
                        strokeWidth=stroke_width,
                    ))
            if HIGHLIGHT_SELECTED_CONNECTIONS:
                for source_x, source_y, target_x, target_y, other_node, stroke_color in \
                        self.ngs_hash_to_node_to_list_of_connections.get(ngs_hash, {}).get(node_name, []):
                    connection_is_active = check_if_connection_is_active_for_this_role_within_node(node_name, other_node)
                    connection_id = f'highlight_connection__{ngs_hash}__{node_name}__{other_node}'
                    connection_id = connection_id.replace('.', '_')
                    graph_overlay_elements.append(dash_svg.Line(
                        id=connection_id,
                        x1=str(source_x), x2=str(target_x), y1=str(source_y), y2=str(target_y),
                        stroke=stroke_color,
                        strokeWidth=3 if connection_is_active else 1,
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
                ('batch_aggregation', batch_aggregation_value),
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
                cleaned_tensor_name = self.clean_name_of_tensor_or_node(
                    'tensor', node.type_of_tensor, node.full_unique_name[len('node__'):],
                )
                cleaned_role_within_tensor = self.clean_name_of_tensor_or_node(
                    'role', node.type_of_tensor, role_of_tensor_in_node_value,
                )
                children = [
                    html.Table(
                        [html.Tr([html.Th(col) for col in
                                  ['Trial', 'Training Step', 'Iteration',
                                   'Step Type', 'Batch',
                                   'Node', 'Role',
                                   'Tensor Type', 'Tensor Shape',
                                   ]])] +
                        [html.Tr([
                            html.Td(val) for val in [
                                trials_value, training_step_value, iteration_value,
                                tsc.type_of_execution,  # This may not equal type_of_execution, which can be "any_value" because it's a filter
                                batch_aggregation_value,
                                cleaned_tensor_name,
                                cleaned_role_within_tensor,
                                node.type_of_tensor,
                                f"[{', '.join([str(a) for a in tensor_shape])}]",
                            ]
                        ])],
                        id='selected-values-summary-table',
                    ),
                    html.Table([html.Tr([html.Th(col) for col in ['Item', '', '', 'value']])] + rows),
                ]
            elif display_type_radio_buttons == 'Network':
                hide_containers_for_tensors = True
                children = [
                    html.Div(f"{self.get_formatted_overview_of_module_parameters(tsc)}", className="metadata-div"),
                ]
            elif display_type_radio_buttons == 'Notes':
                hide_containers_for_tensors = True
                children = [
                    html.Div('\n'.join(self.get_notes_for_trial(trials_value)), className="metadata-div"),
                ]
            elif display_type_radio_buttons == 'Graphs':
                hide_containers_for_tensors = True
                children = [
                    html.Div(self.display_kpi_graphs(trials_value), className="kpi-graphs-div"),
                ]
            elif display_type_radio_buttons == 'Visualization':
                hide_containers_for_tensors = False
                children = [
                    html.Div(self.create_external_visualization(
                        recordings, type_of_execution,  # Note that type_of_execution can be "any_value" because it's a filter
                        tsc, ngs, db,
                        training_step_value, type_of_recording_value, batch_aggregation_value, iteration_value, node_name,
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
        tsc, recordings = self.get_training_step_configuration_and_recordings(trials_value, training_step_value, type_of_execution)
        # Special case: If nothing was clicked, just return the previous_name_of_selected_node
        if max([-1 if a is None else a for a in
                navigation_button_clicks + clicks_per_node]) <= self.last_navigation_click_event_time:
            return previous_name_of_selected_node, False
        # Identify which Nodes exist in the selected NodeGraphStructure
        graph_configuration = tsc.graph_configuration_per_iteration[iteration_value]
        ngs_hash = graph_configuration.hash_of_node_graph_structure
        ngs = self.ngs_hash_to_ngs[ngs_hash]
        node_to_dash_id = self.ngs_hash_to_node_to_dash_id[ngs_hash]
        dash_id_to_node = {
            dash_id: node for node, dash_id in node_to_dash_id.items()
        }
        assert len(dash_id_to_node) == len(node_to_dash_id)
        names = [
            dash_id_to_node.get(dash_id, None) for dash_id in self.list_of_unique_node_dash_ids
        ]
        names_that_exist_in_the_selected_configuration = {
            node for node in ngs.name_to_node.keys()
        }
        assert len(clicks_per_node) == len(self.list_of_unique_node_dash_ids), (len(clicks_per_node), len(self.list_of_unique_node_dash_ids))
        assert len(names) == len(self.list_of_unique_node_dash_ids), (len(names), len(self.list_of_unique_node_dash_ids))
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
            grid_of_nodes = self.ngs_hash_to_grid_of_nodes[ngs_hash]
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
        return name_of_selected_node, True

    @utilities.runtime_analysis_decorator
    def get_formatted_overview_of_module_parameters(self, tsc: TrainingStepConfiguration):

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

        total_parameters, lines = rec(tsc.modules_and_parameters, 0)
        res = f"Total parameters: {total_parameters:18,d}\n" + "\n".join(lines)
        return res

    def clean_name_of_tensor_or_node(self, obj_type, node_type, text):
        if obj_type == 'tensor':
            pass
        elif obj_type == 'node':
            assert text.startswith("node__")
            text = text[len("node__"):]
        elif obj_type == 'role':
            pass
        else:
            assert False, obj_type
        # For parameters, remove the suffix.
        # It will always be present, so there is no point in displaying it at all
        if node_type == 'parameter':
            if obj_type in ['tensor', 'node']:
                assert text.endswith(SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS)
                text = text[:-len(SUFFIX_TO_AVOID_DUPLICATES_WHEN_REUSING_REFERENCES_FROM_OLDER_ITERATIONS)]
            elif obj_type == 'role':
                pass
        return text

    @utilities.runtime_analysis_decorator
    def get_notes_for_trial(self, trial_id):
        path = self.path / 'trials' / str(trial_id) / 'notes.json'
        if path.exists():
            with open(path, 'r') as f:
                notes = json.load(f)
            return notes
        return ["No Notes have been recorded for this trial."]

    @utilities.runtime_analysis_decorator
    def display_kpi_graphs(self, trials_value):
        if trials_value not in self.trial_to_kpi_graph_excerpt:
            self.load_kpi_graphs_for_trial(trials_value)
        graph_excerpt = self.trial_to_kpi_graph_excerpt.get(trials_value, None)
        if graph_excerpt is None:
            return html.Div(children=["No graphs have been created. Use record_kpi_in_graph() to create graphs."])
        #
        # Determine colors to use for the graphs
        #
        colors_by_type_of_execution = {}
        colors_by_kpi_name = {}
        for kpi_group, a in graph_excerpt.items():
            for type_of_execution, b in a.items():
                for kpi_name, stats in b.items():
                    colors_by_type_of_execution[type_of_execution] = True
                    colors_by_kpi_name[kpi_name] = True
        colors_by_type_of_execution = utilities.map_to_distinct_colors(colors_by_type_of_execution)
        colors_by_kpi_name = utilities.map_to_distinct_colors(colors_by_kpi_name)
        #
        # Draw the graphs
        #
        graphs = []
        for kpi_group, a in graph_excerpt.items():
            plots = []
            for type_of_execution, b in a.items():
                for kpi_name, stats in b.items():
                    vals = stats['vals']
                    xs = []
                    ys = []
                    for val in vals:
                        x = val['timepoint']
                        y = val['val']
                        xs.append(x)
                        ys.append(y)
                    name = f"{kpi_name}__{type_of_execution}" if kpi_name else type_of_execution
                    plots.append(go.Scatter(
                        x=xs, y=ys,
                        name=name,
                        mode="lines+markers",
                        textposition="bottom center",
                        text=name,
                        marker=dict(color=colors_by_type_of_execution[type_of_execution]),
                        line=dict(color=colors_by_kpi_name[kpi_name]),
                    ))
            fig = go.Figure(data=plots, layout=go.Layout(
                title=f'{kpi_group}',
                xaxis=dict(title='step'),
                yaxis=dict(title='value'),
                legend=dict(),
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
        if len(files) > 0:
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
