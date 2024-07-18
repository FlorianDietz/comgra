import dash
from dash import dcc, html
import dash_svg
import plotly.graph_objs as go

from comgra.objects import TrainingStepConfiguration, NodeGraphStructure
from comgra.utilities import PseudoDb


def create_visualization(
        recordings, type_of_execution, tsc: TrainingStepConfiguration, ngs: NodeGraphStructure, db: PseudoDb,
        training_step_value, type_of_recording_value, batch_aggregation_value, iteration_value,
        node_name, role_of_tensor_in_node_value,
):
    #
    # Comgra uses https://github.com/plotly/dash for visualization.
    # This function needs to return a dash element.
    #
    messages = [f"This is a demonstration of custom visualization, using the file 'example_custom_visualization.py'.\n"]
    #
    # Visualize helper_partial_sums in a graph
    #
    if node_name == 'node__helper_partial_sums':
        # Get helper_partial_sums
        # We use some selectors as filters, and leave other selectors variable so that we get all the data we want
        num_iterations_on_the_selected_training_step = len(tsc.graph_configuration_per_iteration)
        filters = {
            'training_step': training_step_value,
            'type_of_tensor_recording': 'forward',
            'batch_aggregation': batch_aggregation_value,
            # 'iteration': iteration_value,
            'node_name': 'node__helper_partial_sums',
            # 'role_within_node': role_of_tensor_in_node_value,
            'record_type': 'data',
            'item': 'neuron',
            # 'metadata': None,
        }
        list_of_matches, _ = db.get_matches(filters)
        relevant_data = [
            # Each entry of list_of_matches returns a single number, and a tuple describing that number.
            # The format of list_of_matches is as shown here.
            # The metadata field refers to the type of 'item' selected in the filters,
            # and in our case specifies the index of the neuron.
            (
                # Extract the iteration, the neuron, and the corresponding value
                num_iterations_on_the_selected_training_step - 1 if role_of_tensor_in_node_value == 'final'
                else int(role_of_tensor_in_node_value[len('up_to_iteration_'):]),
                int(metadata),
                result_value
            )
            for (training_step_value, type_of_recording_value, batch_aggregation_value, iteration_value, name_of_selected_node, role_of_tensor_in_node_value, record_type, item, metadata), result_value
            in list_of_matches
        ]
        num_neurons = 1 + max(neuron for iteration, neuron, val in relevant_data)
        assert len(relevant_data) == num_iterations_on_the_selected_training_step * num_neurons, \
            "This query should return one entry per recorded neuron and iteration"
        # Create a graph to visualize helper_partial_sums
        plots = []
        for i in range(num_neurons):
            xys = sorted([(iteration, val) for iteration, neuron, val in relevant_data if neuron == i], key=lambda x: x[0])
            xs = [iteration for iteration, val in xys]
            ys = [val for iteration, val in xys]
            name = f"neuron_{i}"
            plot = go.Scatter(
                x=xs, y=ys,
                name=name,
                text=name,
            )
            plots.append(plot)
        fig = go.Figure(data=plots, layout=go.Layout(
            title="helper_partial_sums",
            xaxis=dict(title="iteration"),
            yaxis=dict(title="value"),
            legend=dict(x=0, y=1),  # Positioning of the legend (0 to 1, where 0 is left/bottom and 1 is right/top)
            hovermode='closest'
        ))
        example_graph = dcc.Graph(figure=fig)
    else:
        # If the node node__helper_partial_sums is not selected, we just show an error message instead.
        # We could just always show the graph regardless of the selected node, but there is a minor problem:
        # The batch_aggregation_value depends on the user's current selection, and it could have a value that is invalid
        # for the helper_partial_sums, which would make it ambiguous what we are supposed to display.
        # (in particular, if a node that represents a network parameter is selected, then batch_aggregation_value will
        #  be invalid, because network parameters don't have a batch dimension.)
        # To prevent any confusion, we just display a message in this case.
        example_graph = "Select the node helper_partial_sums to see a visualization for it\n"
    messages.append(example_graph)
    #
    # Print some information about the current selection
    #
    filters = {
        # Note: The 'recordings' are not used in the filters because comgra uses a separate database object
        # for each recording, so this filter is already covered implicitly
        'training_step': training_step_value,
        'type_of_tensor_recording': type_of_recording_value,
        'batch_aggregation': batch_aggregation_value,
        'iteration': iteration_value,
        'node_name': node_name,
        'role_within_node': role_of_tensor_in_node_value,
        'record_type': 'data',
        # 'item': None,
        # 'metadata': None,
    }
    list_of_matches, possible_attribute_values = db.get_matches(filters)
    messages.append(f"filters:\n{filters}\n")
    # list_of_matches gives all the values in the database that match our data
    # See the visualization above for the format of the returned data.
    messages.append(f"number of matches: {len(list_of_matches)}:\n")
    for match in list_of_matches:
        messages.append(f"{match}\n")
    # possible_attribute_values contains alternative values that result in at least one valid entry in the database
    # if all other values are kept unchanged.
    # This is mostly relevant for the GUI to define which selector values are valid
    # and is not so important for visualization.
    messages.append(f"possible_attribute_values:\n{possible_attribute_values}\n")
    #
    # Combine and return the outputs
    #
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
