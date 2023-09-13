import shutil
from pathlib import Path

import argparse

from comgra.recorder import ComgraRecorder
from comgra.objects import DecisionMakerForRecordingsFrequencyPerType

import torch
import torch.nn as nn

# Comgra uses a ComgraRecorder object that needs to be available in all code blocks where you want to record something.
# In this example, I just made it a global variable, but it's better to pass it through to your code in arguments.
COMGRA_RECORDER = None


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # Tensors in comgra need to get unique names in order to be recorded correctly.
        # This nn.Module gets instantiated twice.
        # To ensure the name is unique regardless, comgra provides a helper function to get the name
        # of the current module, as recorded by track_module()
        COMGRA_RECORDER.register_tensor(f"{COMGRA_RECORDER.get_name_of_module(self)}__hidden_state", x, recording_type='neurons')
        x = self.fc2(x)
        return x


# Our model uses several identical modules with different names.
# It also has a hidden state that will also be recorded in comgra.
class CompleteModule(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, output_size):
        super(CompleteModule, self).__init__()
        self.subnet_pre = NeuralNet(input_size + memory_size, hidden_size, hidden_size)
        self.subnet_out = NeuralNet(hidden_size, hidden_size, output_size)
        self.subnet_mem = NeuralNet(hidden_size, hidden_size, memory_size)

    def forward(self, x):
        pre = self.subnet_pre(x)
        out = self.subnet_out(pre)
        memory = self.subnet_mem(pre)
        return out, memory

def run_demonstration(comgra_root_path, comgra_group):
    global COMGRA_RECORDER
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Hyper-parameters
    batch_size = 50
    num_iterations = 3
    num_training_steps = 10000
    # Generate some fake data for training
    # This data is meaningless. It's used here to avoid having to download external data for the demonstration.
    # This data has higher dimensionality than number of datapoints, which means it should be learnable.
    # Inspecting the results in the comgra visualization later should show
    # that the loss goes down as the training proceeds.
    # To see this, select iteration 2 in the UI, click on the Node that represents the loss (on the very right)
    # and then check how the value changes as you move the slider for the training step.
    num_datasets = batch_size
    input_size = 100
    memory_size = 50
    hidden_size = 200
    output_size = 100
    all_input_data = torch.rand((num_datasets, input_size)).to(device)
    all_target_data = torch.rand((num_datasets, input_size)).to(device)
    # Model
    model = CompleteModule(input_size, hidden_size, memory_size, output_size).to(device)
    # Loss and Optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    # Delete previous runs of comgra
    shutil.rmtree(comgra_root_path / comgra_group, ignore_errors=True)
    # Initialize comgra
    # This command tells comgra to permanently register all parameters of a given module.
    COMGRA_RECORDER = ComgraRecorder(
        # The root folder for all comgra data
        comgra_root_path=comgra_root_path,
        # All runs of comgra that share the same 'group' will be loaded in the same application
        # When you run the server.py application by calling "comgra" from the commandline,
        # the last folder in the --path argument will be the group name.
        # The different trials in a group can then be selected by their 'trial_id' and compared.
        group=comgra_group, trial_id='example_trial',
        # These parameters can be left empty, but it is recommended to fill them in
        # if your computational graph is complex.
        # They ensure that similar module parameters get grouped together visually.
        # All module parameters whose complete name (including the list of names of modules they are contained in)
        # match one of these prefixes are grouped together.
        # This can also be used to group tensors together that have a variable number of occurrences,
        # for example the elements of an attention mechanism.
        prefixes_for_grouping_module_parameters_visually=[
            'root_module.subnet_pre',
            'root_module.subnet_out',
            'root_module.subnet_mem',
        ],
        prefixes_for_grouping_module_parameters_in_nodes=[],
        # Parameters that will be recorded in a JSON file, to help you with comparing things later.
        parameters_of_trial={},
        # How often do you want comgra to make a recording?
        # There are several options for this.
        # This one is often the most effective one:
        # At the beginning of each training step later in this code, you specify what the type of this recording is.
        # For example, you could differentiate between randomly selected training and training on a specific example
        # that you would like to inspect in more detail.
        # This recording type ensures that a recording is made if the last training of the specified type was at least
        # N training steps ago.
        # In this way, you make sure that each type gets recorded often enough to be useful,
        # but not so often that the program slows down and your hard drive gets filled up.
        decision_maker_for_recordings=DecisionMakerForRecordingsFrequencyPerType(min_training_steps_difference=1000),
        # Comgra records data both in terms of statistics over the batch dimension and in terms of
        # individual items in the batch.
        # If batches are large, this consumes too much memory and slows down the recording.
        # This number tells comgra only to record the first N items of each batch.
        # Note that the statistics over the batch that also get recorded are still calculated over the whole batch.
        max_num_batch_size_to_record=5,
        # Use this to turn comgra off throughout your whole project.
        comgra_is_active=True,
        # A performance parameter you can experiment with if comgra is too slow.
        # If this is too low, comgra becomes slow.
        # If this is too high, the program may crash due to memory problems.
        # (This problem is caused by a serialization bug in a backend library.)
        max_num_mappings_to_save_at_once_during_serialization=10000,
        # An optional feature to skip the recording of KPIs that are particularly expensive to calculate.
        calculate_svd_and_other_expensive_operations_of_parameters=True,
    )
    # Register the modules you are using.
    # This recursively goes through all contained modules and all their weight parameters
    # and registers them to be recorded later.
    COMGRA_RECORDER.track_module("root_module", model)
    #
    # Train the model
    #
    for training_step in range(num_training_steps):
        # Each time a new training step or epoch is started, call this function to inform comgra of this.
        # It will automatically decide whether to skip making a recording based on
        # decision_maker_for_recordings and override__recording_is_active
        COMGRA_RECORDER.start_next_recording(
            training_step, batch_size,
            # Recordings will be skipped when not in training mode as there is no computational graph
            is_training_mode=True,
            # This is the string that is used by DecisionMakerForRecordingsFrequencyPerType
            # to decide what type of thing is being recorded here.
            # Here, we just record even and odd-numbered epochs separately as an example.
            # You will be able to filter by 'even' and 'odd' in the visualization later.
            type_of_execution_for_diversity_of_recordings='even' if training_step % 2 == 0 else 'odd',
            # The default value for record_per_batch_index of register_tensor() calls.
            # Setting this to True means that individual examples will be stored, up to max_num_batch_size_to_record,
            # in addition to statistics across the batch.
            # This can be useful if you want to see the full details for a particular training step.
            record_all_tensors_per_batch_index_by_default=True,
            # None = use decision_maker_for_recordings to decide whether to record (default)
            # True = record
            # False = don't record
            override__recording_is_active=None,
        )
        # For this demonstration, the dataset size equals the batch_size,
        # and all data is used for training on every iteration.
        x = all_input_data
        target = all_target_data
        # Initialize the memory
        memory = torch.zeros((batch_size, memory_size), device=device)
        # Iterate
        for iteration in range(num_iterations):
            # Tell comgra that an iteration has started.
            # You can start recording tensors after this, with register_tensor().
            COMGRA_RECORDER.start_forward_pass(
                # This is a string that needs to uniquely identify the shape of the computational graph.
                # The computational graph usually looks different on the iteration where it receives a loss
                # than on any preceding iteration.
                # Depending on your use case, other parameters may influence what the computational graph looks like.
                # The recommended approach is to just concatenate a string here that is built from all the factors
                # that influence what your computation graph looks like.
                # Comgra will create a separate configuration for each unique value of this parameter,
                # so it can slow down if you go overboard with this and create more unique values than necessary.
                # If you assign the same configuration_type to two calls with a different graph,
                # you will receive an error message.
                configuration_type=f"{'loss' if iteration == num_iterations - 1 else 'no_loss'}"
            )
            # Record the inputs of the iteration.
            # This includes both the actual input 'x' and the state variable 'memory'.
            # (From the perspective of one iteration only, the memory is an input.)
            # We are going to record a value for each individual neuron
            # instead of just statistical information over the whole tensor.
            # In real applications, whether you want that level of granularity will depend on your use case.
            # For the 'x' tensor we are skipping that part,
            # so that you can see the difference in the visualization later.
            # Note that each register_tensor() call uses a different type: is_input, is_output, etc.
            # Each of these will be displayed differently in the visualization.
            COMGRA_RECORDER.register_tensor(f"x_in", x, is_input=True)
            COMGRA_RECORDER.register_tensor(f"memory_in", memory, is_input=True, recording_type='neurons')
            x = torch.cat([x, memory], dim=1)
            # Forward pass
            output, memory = model(x)
            x = output
            COMGRA_RECORDER.register_tensor(f"x_out", x, is_output=True, recording_type='neurons')
            COMGRA_RECORDER.register_tensor(f"memory_out", memory, is_output=True, recording_type='neurons')
            assert output.shape == (batch_size, output_size)
            assert memory.shape == (batch_size, memory_size)
            # Apply the loss on the last iteration only
            if iteration == num_iterations - 1:
                # Tell comgra that we are now performing a backward pass
                COMGRA_RECORDER.start_backward_pass()
                COMGRA_RECORDER.register_tensor(f"target", target, is_target=True, recording_type='neurons')
                loss = criterion(output, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                COMGRA_RECORDER.register_tensor(f"loss", loss, is_loss=True)
                # This command causes comgra to record all losses that are currently on the module parameters,
                # for each module parameter registered through track_module()
                COMGRA_RECORDER.record_current_gradients(f"gradients")
                if training_step % 100 == 0:
                    print(f"Epoch {training_step}: Loss = {loss.item()}")
            # Tell comgra that the iteration has finished.
            # At this point, you can specify whether you want to run a sanity check to make sure that you specified
            # configuration_type of start_forward_pass() correctly.
            # In this example, we only do this for the first few hundred steps, as it costs extra time to compute.
            # If you skip this sanity check, you might not realize that you are recording two different computational
            # graphs under the same name, and this will lead to errors in the visualization later.
            COMGRA_RECORDER.finish_iteration(sanity_check__verify_graph_and_global_status_equal_existing_file=training_step < 500)
            # Comgra will raise an exception if the same tensor is registered twice under different names.
            # This is a feature that should help you catch errors, but if you plan to register
            # the output of one iteration again as the input on the next iteration, you will get an error
            # even though what you are doing is intentional.
            # To prevent this, just multiply all tensors that will be reused on the next iteration with 1.
            x = x * 1
            memory = memory * 1
        # Finish a batch.
        # This is the counterpart to start_next_recording
        COMGRA_RECORDER.finish_batch()
    # Test the model
    print("This script does not include any tests. "
          "Comgra can only be used during training, not during testing, "
          "because pytorch does not generate a computational graph during testing.")


def main():
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--path', dest='path', default=None)
    parser.add_argument('--group', dest='group', default=None)
    args = parser.parse_args()
    if args.path is None:
        args.group = 'publication_test'
        args.path = (Path(__file__).parent.parent.parent / 'testing_data').absolute()
        args.path.mkdir(exist_ok=True, parents=True)
    path = Path(args.path).absolute()
    assert path.exists(), path
    run_demonstration(path, args.group)


if __name__ == '__main__':
    main()

