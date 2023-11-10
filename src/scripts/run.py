import argparse
from pathlib import Path
import shutil

import comgra
from comgra.recorder import ComgraRecorder
from comgra.objects import DecisionMakerForRecordingsFrequencyPerType

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class Demonstration:
    def __init__(self, comgra_root_path, comgra_group):
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Delete previous runs of comgra. This is optional.
        self.comgra_root_path = comgra_root_path
        self.comgra_group = comgra_group
        shutil.rmtree(self.comgra_root_path / self.comgra_group, ignore_errors=True)
        # Constants and fields for later use
        self.batch_size = 128
        self.memory_size = 50
        self.input_size = 5
        self.hidden_size = 200
        self.output_size = 5
        self.current_configuration = None
        self.model = None
        self.criterion = None
        self.optimizer = None
        # Generate some synthetic data for training and testing.
        self.task_data = self.generate_task_data()

    def run_all_configurations(self):
        # This example script trains and tests two different configurations of the network.
        configurations = [
            'bugged_original_version',
            'no_activation_function_on_output_layer',
        ]
        for configuration in configurations:
            self.current_configuration = configuration
            self.run_current_configuration()

    def run_current_configuration(self):
        # Model, Criterion, Optimizer
        self.model = CompleteModule(
            self.input_size, self.hidden_size, self.memory_size, self.output_size
        ).to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # Initialize comgra
        # This command tells comgra to permanently register all parameters of a given module.
        # Note:
        # Comgra uses a ComgraRecorder object that needs to be available in all code blocks where
        # you want to access comgra.
        # (Comgra is object-based instead of static
        # so that you can use multithreading and record several runs in parallel.)
        # If you don't want to pass a new parameter for this through all your modules,
        # and you are not using multithreading anyway,
        # then you can just store this in a global variable inside the comgra library itself for simplicity,
        # as is done here.
        comgra.my_recorder = ComgraRecorder(
            # The root folder for all comgra data
            comgra_root_path=self.comgra_root_path,
            # All runs of comgra that share the same 'group' will be loaded in the same application
            # When you run the server.py application by calling "comgra" from the commandline,
            # the last folder in the --path argument will be the group name.
            # The different trials in a group can then be selected by their 'trial_id' and compared.
            group=self.comgra_group,
            trial_id=f'trial_{self.current_configuration}',
            # These parameters can be left empty, but it is recommended to fill them in
            # if your computation graph is complex.
            # They ensure that similar module parameters get visually grouped together in the dependency graph.
            # All module parameters whose complete name (including the list of names of modules they are contained in)
            # match one of these prefixes are grouped together.
            prefixes_for_grouping_module_parameters_visually=[
                'root_module.subnet_pre',
                'root_module.subnet_out',
                'root_module.subnet_mem',
            ],
            prefixes_for_grouping_module_parameters_in_nodes=[],
            # This parameter determines when and how often Comgra makes a recording.
            # There are several options for this.
            # This one is often the most effective one:
            # At the beginning of each training step, later in this code, you specify what the type of this recording is.
            # For example, you could differentiate between randomly selected training and training on a specific example
            # that you would like to inspect in more detail.
            # The DecisionMakerForRecordingsFrequencyPerType recording type ensures that a recording is made
            # if the last training of the specified type was at least N training steps ago.
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
            # An optional parameter you can experiment with if comgra is too slow.
            # If this is too low, comgra becomes slow.
            # If this is too high, the program may crash due to memory problems.
            # (This problem is caused by a serialization bug in a backend library.)
            max_num_mappings_to_save_at_once_during_serialization=10000,
            # An optional feature to skip the recording of statistics that are particularly expensive to calculate.
            calculate_svd_and_other_expensive_operations_of_parameters=True,
        )
        # Register the modules you are using.
        # This recursively goes through all contained modules and all their weight parameters
        # and registers them so that they get recorded along with the other tensors.
        comgra.my_recorder.track_module("root_module", self.model)
        # Record information about the training and testing data in comgra
        comgra.my_recorder.add_note(f"Total training data: {sum([len(dataloader.dataset) for _, use_for_training, dataloader in self.task_data if use_for_training]):,.0f}")
        comgra.my_recorder.add_note(f"Total test data: {sum([len(dataloader.dataset) for _, use_for_training, dataloader in self.task_data if not use_for_training]):,.0f}")
        for i, (num_iterations, use_for_training, dataloader) in enumerate(self.task_data):
            comgra.my_recorder.add_note(f"Dataset {i}: {num_iterations} iterations, used for training: {use_for_training}, {len(dataloader.dataset):,.0f}")
        # Train the model
        num_training_steps = 20_000
        for training_step in range(num_training_steps):
            # Pick a random dataloader and run a batch of it.
            # If its number of iterations is above 10, then do not train on it.
            # We consider these cases out of distribution in our example,
            # and we are interested in what comgra records about it.
            idx = training_step % len(self.task_data)
            num_iterations, use_for_training, dataloader = self.task_data[idx]
            self.run_one_training_step(training_step, num_iterations, dataloader, use_for_training)


    def run_one_training_step(self, training_step, num_iterations, dataloader, update_the_model_parameters):
        sample = next(iter(dataloader))
        input_tensor = sample['input']
        target_tensor = sample['target']
        batch_size = self.batch_size
        assert input_tensor.shape == (batch_size, num_iterations, 5), input_tensor.shape
        assert target_tensor.shape == (batch_size, 5), target_tensor.shape
        # Each time a new training step is started, call this function to inform comgra of this.
        # It will automatically decide whether to make a recording based on
        # decision_maker_for_recordings and override__recording_is_active
        comgra.my_recorder.start_next_recording(
            training_step,
            batch_size,
            # This is the string that is used by DecisionMakerForRecordingsFrequencyPerType
            # to decide what type of thing is being recorded here.
            # The string depends on the number of iterations, which will ensure that recordings for each possible
            # number of iterations are made regularly, even if some of them occurred less often than others.
            type_of_execution=f'{num_iterations:02d}_iterations',
            # If this is set to False, only statistics are stored by default whenever you call register_tensor() later,
            # which means less data to be stored and processed.
            # You can still use record_per_batch_index=True with register_tensor() for the tensors for
            # which you want to see all details.
            record_all_tensors_per_batch_index_by_default=True,
            # None = use decision_maker_for_recordings to decide whether to record (default)
            # True = record
            # False = don't record
            # Important: Set this to False when you are running the network in evaluation mode,
            # as torch will not generate any computation graphs in this case, so comgra can't work.
            override__recording_is_active=None,
        )
        # Initialize the memory
        memory = torch.zeros((batch_size, self.memory_size), device=self.device)
        # Iterate
        for iteration in range(num_iterations):
            input_for_this_iteration = input_tensor[:, iteration, :]
            # Tell comgra that an iteration has started.
            # You can start recording tensors after this, with register_tensor().
            comgra.my_recorder.start_iteration(
                # This is a string that needs to uniquely identify the shape of the dependency graph.
                # The dependency graph usually looks different on the iteration where it receives a loss
                # than on any preceding iteration.
                # Depending on your use case, other parameters may influence what the dependency graph looks like.
                # The recommended approach is to just concatenate a string here that is built from all the factors
                # that influence what your dependency graph looks like.
                # Comgra will create a separate configuration for each unique value of this parameter,
                # so it can slow down if you go overboard with this and create more unique values than necessary.
                # If you assign the same configuration_type to two calls with a different graph,
                # you will receive an error message.
                configuration_type=f"{'has_loss' if iteration == num_iterations - 1 else 'no_loss'}"
            )
            # Record the inputs of the iteration.
            # This includes both the actual input 'input_for_this_iteration' and the state variable 'memory'.
            # (From the perspective of one iteration only, the memory is an input.)
            #
            # Note that each register_tensor() call uses a different type: is_input, is_output, etc.
            # Each of these will be colored differently in the visualization.
            #
            # By default, this records the values of all neurons in the tensor,
            # which can be costly for performance if the tensor is very large.
            # You can also specify recording_type='kpis' to record only KPIs instead of all neurons.
            # KPIs are mean(), abs().mean(), std(), and abs().max()
            comgra.my_recorder.register_tensor(f"input", input_for_this_iteration, is_input=True)
            comgra.my_recorder.register_tensor(f"memory_in", memory, is_input=True)
            x = torch.cat([input_for_this_iteration, memory], dim=1)
            # Forward pass
            output, memory = self.model(x)
            # Apply a sigmoid to the outputs
            # (This is not necessarily the smartest idea, a softmax is better.
            # But it's useful for illustrative purposes in this example.)
            output = torch.sigmoid(output)
            comgra.my_recorder.register_tensor(f"output", output, is_output=True)
            comgra.my_recorder.register_tensor(f"memory_out", memory, is_output=True)
            assert output.shape == (batch_size, self.output_size)
            assert memory.shape == (batch_size, self.memory_size)
            # Apply the loss on the last iteration only
            if iteration == num_iterations - 1:
                # Tell comgra that we are now performing a backward pass and register some more tensors
                comgra.my_recorder.start_backward_pass()
                comgra.my_recorder.register_tensor(f"target", target_tensor, is_target=True)
                # We calculate and register some helper tensors:
                # The partial sums of the inputs up to some iteration.
                # The Node 'helper_partial_sums' in the visualization will have several different values,
                # selectable through the dropdown "Role of tensor".
                # Having information like this easily accessible
                # right next to the target in the GUI can help you with debugging.
                for i in range(0, num_iterations):
                    helper_partial_sums = input_tensor[:, :i+1, :].sum(dim=1).detach()
                    comgra.my_recorder.register_tensor(
                        f"helper_partial_sums_up_to_iteration_{i}", helper_partial_sums, is_target=True,
                        node_name=f"helper_partial_sums", role_within_node=f"up_to_iteration_{i}"
                    )
                # Calculate the loss and perform a backward pass as normal
                loss = self.criterion(output, target_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                comgra.my_recorder.register_tensor(f"loss", loss, is_loss=True)
                accuracy = (output.argmax(dim=1).eq(target_tensor.argmax(dim=1))).float().mean()
                # We skip the update step for some data, because we want to see how that affects network values.
                # (We unfortunately can not use torch.no_grad() if we want to record data in comgra,
                # because the computation graph needs to be calculated for comgra to work.)
                if update_the_model_parameters:
                    self.optimizer.step()
                # This command causes comgra to record all losses that are currently on the module parameters,
                # for each module parameter registered through comgra.my_recorder.track_module()
                comgra.my_recorder.record_current_gradients(f"gradients")
                # We can make use of comgra's smart decision-making which training steps to record for our own logging
                if comgra.my_recorder.recording_is_active():
                    note = ""
                    if num_iterations == 1:
                        note = "-" * 20 + "\n"
                    note += (f"{self.current_configuration}, Step {training_step:5}: "
                             f"{num_iterations:2} iterations "
                             f"{'(training)' if update_the_model_parameters else ' (testing)'}  -  "
                             f"Loss = {loss.item():10.6f}  -  "
                             f"Accuracy on batch = {accuracy:2.3f}")
                    # Make a quick note of the console output in comgra as well,
                    # just so the information is all in one place.
                    # This feature is also helpful for recording debug messages.
                    print(note)
                    comgra.my_recorder.add_note(note)
            # Tell comgra that the iteration has finished.
            comgra.my_recorder.finish_iteration()
            # We multiply a tensor with 1 here to prevent an issue:
            # As an error-catching feature, Comgra will raise an exception if the same tensor is
            # registered twice under different names.
            # But if you plan to register the output of one iteration again as an input on the next iteration,
            # you will get an error even though what you are doing is intentional.
            # To prevent this, just multiply all tensors that will be reused on the next iteration with 1.
            memory = memory * 1
        # Finish a batch.
        # This is the counterpart to start_next_recording.
        # All tensors registered by comgra will be serialized at this point.
        comgra.my_recorder.finish_recording()

    def generate_task_data(self):
        # Generate some datasets for training (lengths 1 to 10) and for testing (lengths 15 and 20)
        task_data = [
            (number_of_iterations, number_of_iterations <= 10, DataLoader(
                ExampleDataset(device=self.device, dataset_size=10_000, number_of_iterations=number_of_iterations),
                batch_size=self.batch_size,
                shuffle=True,
            ))
            for number_of_iterations in list(range(1, 11)) + [15, 20]
        ]
        return task_data


class ExampleDataset(Dataset):

    def __init__(self, device, dataset_size, number_of_iterations):
        self.dataset_size = dataset_size
        self.number_of_iterations = number_of_iterations
        self.data = []
        # Inputs are sequences of 5 random numbers in [0;1]
        # Targets are 1 for the sequence with the greatest sum and 0 for the others.
        self.inputs = torch.rand((dataset_size, number_of_iterations, 5)).to(device)
        tmp = self.inputs.sum(dim=1)
        self.targets = tmp.eq(tmp.max(dim=1, keepdim=True)[0].expand(dataset_size, 5)).float()

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        sample = {
            'input': self.inputs[idx, :],
            'target': self.targets[idx],
        }
        return sample


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.fnn1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.LeakyReLU(negative_slope=0.01)
        self.fnn2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fnn1(x)
        x = self.activation(x)
        # Tensors in comgra need to get unique names in order to be recorded correctly.
        # This nn.Module will be instantiated twice, so its name is not unique.
        # To ensure the name is unique regardless, comgra provides a helper function to get the name
        # of the current module and all its parents, as recorded by comgra.my_recorder.track_module()
        comgra.my_recorder.register_tensor(f"{comgra.my_recorder.get_name_of_module(self)}__hidden_state", x)
        x = self.fnn2(x)
        if DEMONSTRATION.current_configuration != 'no_activation_function_on_output_layer':
            x = self.activation(x)
        comgra.my_recorder.register_tensor(f"{comgra.my_recorder.get_name_of_module(self)}__out", x)
        return x * 1


# Our model uses several identical modules with different names, for illustration purposes.
# It also has hidden states that will also be recorded in comgra.
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


# A hacky global helper variable, to make it easier to pass parameters to the modules
DEMONSTRATION: Demonstration

def main():
    global DEMONSTRATION
    parser = argparse.ArgumentParser(description="Run an experiment.")
    parser.add_argument('--path', dest='path', default=None)
    parser.add_argument('--group', dest='group', default=None)
    args = parser.parse_args()
    if args.path is None:
        args.group = 'testcase_for_demonstration'
        args.path = (Path(__file__).parent.parent.parent / 'testing_data').absolute()
        args.path.mkdir(exist_ok=True, parents=True)
    path = Path(args.path).absolute()
    assert path.exists(), path
    DEMONSTRATION = Demonstration(path, args.group)
    DEMONSTRATION.run_all_configurations()


if __name__ == '__main__':
    main()

