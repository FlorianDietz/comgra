import argparse
from pathlib import Path
import shutil

# Activate debug mode
from comgra import utilities_initialization_config
utilities_initialization_config.DEBUG_MODE = True
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
        comgra.my_recorder = ComgraRecorder(
            comgra_root_path=self.comgra_root_path,
            group=self.comgra_group,
            trial_id=f'trial_{self.current_configuration}',
            prefixes_for_grouping_module_parameters_visually=[
                'root_module.subnet_pre',
                'root_module.subnet_out',
                'root_module.subnet_mem',
            ],
            prefixes_for_grouping_module_parameters_in_nodes=[
                'root_module.subnet_pre',
            ],
            decision_maker_for_recordings=DecisionMakerForRecordingsFrequencyPerType(min_training_steps_difference=1000),
            max_num_batch_size_to_record=5,
            comgra_is_active=True,
            calculate_svd_and_other_expensive_operations_of_parameters=True,
        )
        comgra.my_recorder.track_module("root_module", self.model)
        comgra.my_recorder.add_note(f"Total training data: {sum([len(dataloader.dataset) for _, use_for_training, dataloader in self.task_data if use_for_training]):,.0f}")
        comgra.my_recorder.add_note(f"Total test data: {sum([len(dataloader.dataset) for _, use_for_training, dataloader in self.task_data if not use_for_training]):,.0f}")
        for i, (num_iterations, use_for_training, dataloader) in enumerate(self.task_data):
            comgra.my_recorder.add_note(f"Dataset {i}: {num_iterations} iterations, used for training: {use_for_training}, {len(dataloader.dataset):,.0f}")
        # Train the model
        num_training_steps = 3_100
        for training_step in range(num_training_steps):
            # Pick a random dataloader and run a batch of it.
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
        comgra.my_recorder.start_batch(
            training_step,
            batch_size,
            type_of_execution=f'{num_iterations:02d}_iterations',
            record_all_tensors_per_batch_index_by_default=True,
            override__recording_is_active=None,
        )
        memory = torch.zeros((batch_size, self.memory_size), device=self.device)
        comgra.my_recorder.register_tensor(f"initial_memory", memory, node_name='memory', is_initial_value=True)
        for iteration in range(num_iterations):
            input_for_this_iteration = input_tensor[:, iteration, :]
            # Tell comgra that an iteration has started.
            comgra.my_recorder.start_iteration()
            comgra.my_recorder.register_tensor(f"input", input_for_this_iteration, is_input=True)
            x = torch.cat([input_for_this_iteration, memory], dim=1)
            output, memory = self.model(x)
            output = torch.sigmoid(output)
            comgra.my_recorder.register_tensor(f"output", output)
            # memory = memory * 1
            comgra.my_recorder.register_tensor(f"memory_out", memory, node_name='memory')
            assert output.shape == (batch_size, self.output_size)
            assert memory.shape == (batch_size, self.memory_size)
            #
            # Test add_tensor_connection()
            #
            tmp = output.detach()
            comgra.my_recorder.register_tensor(f"unconnected", tmp)
            tmp = output.detach()
            comgra.my_recorder.register_tensor(f"connected_0", tmp) # connect to memory
            comgra.my_recorder.add_tensor_connection(memory, tmp)
            tmp_that_is_not_registered = (output + 1)
            tmp = tmp_that_is_not_registered.detach()
            comgra.my_recorder.register_tensor(f"connected_1", tmp) # connect to output
            comgra.my_recorder.add_tensor_connection(tmp_that_is_not_registered, tmp)
            tmp = output + 1
            comgra.my_recorder.add_tensor_connection(memory, tmp)
            tmp = tmp + 1
            comgra.my_recorder.register_tensor(f"connected_2", tmp) # connect to output + memory
            # This should result in a connection between output and root_module.subnet_mem__out,
            # because memory is a secondary TensorReference for that tensor.
            comgra.my_recorder.add_tensor_connection(output, memory)
            # Detach a tensor but still create a connection
            tmp = output * 1
            tmp = comgra.my_recorder.detach_while_keeping_connection(tmp)
            tmp = tmp * 1
            comgra.my_recorder.register_tensor(f"connected_3", tmp)  # connect to output
            #
            # These lines should result in an error if they are uncommented:
            # The error should show the name of the tensor
            #
            # * self-reference
            # comgra.my_recorder.add_tensor_connection(output, output)
            # * immediate loop
            # comgra.my_recorder.add_tensor_connection(output * 1, output)
            # * longer loop
            # It should give an error even if only one of the two tensors is registered (they may be different errors).
            # It may optional give an error if neither tensor is registered, since then we have a loop that doesn't
            # actually do anything harmful, as it doesn't touch on any tensors.
            # Just be sure the loop doesn't result in a timeout but gets caught
            # and stopped by caching after one iteration.
            # tmp0 = x + 1
            # comgra.my_recorder.register_tensor(f"longer_loop_part_0", tmp0)
            # tmp1 = tmp0 + 1
            # comgra.my_recorder.register_tensor(f"longer_loop_part_1", tmp1)
            # tmp2 = tmp1 + 1
            # comgra.my_recorder.add_tensor_connection(tmp2, tmp0)
            # * A loop that goes all the way back to an input before looping.
            # comgra.my_recorder.add_tensor_connection(output, input_for_this_iteration)
            # * A loop that goes far back, but not quite to an input, and to an unregistered tensor
            # comgra.my_recorder.add_tensor_connection(output, x)
            #
            # Apply the loss on the last iteration only
            #
            if iteration == 2:
                for i in range(0, num_iterations):
                    helper_partial_sums = input_tensor[:, :i+1, :].sum(dim=1).detach()
                    comgra.my_recorder.register_tensor(
                        f"helper_partial_sums_up_to_iteration_{i}", helper_partial_sums,
                        node_name=f"helper_partial_sums", role_within_node=f"up_to_iteration_{i}",
                    )
                    comgra.my_recorder.add_tensor_connection(input_for_this_iteration, helper_partial_sums)  # Not an actual exact dependency, but useful for illustration
                    comgra.my_recorder.register_tensor(
                        f"ref_1_{i}", helper_partial_sums,
                        node_name=f"ref_1", role_within_node=f"up_to_iteration_{i}",
                    )
                    comgra.my_recorder.register_tensor(
                        f"ref_2_{i}", helper_partial_sums,
                        node_name=f"ref_2", role_within_node=f"up_to_iteration_{i}",
                    )
                    comgra.my_recorder.register_tensor(
                        f"helper_copy_{i}", helper_partial_sums*1,
                        node_name=f"helper_copy", role_within_node=f"copy_{i}",
                    )
                helper_copy2 = helper_partial_sums * 1
                comgra.my_recorder.register_tensor(
                    f"helper_copy2", helper_copy2
                )
            if iteration == 3:
                pass
                # Try commenting out some of these. Make sure they show up on the correct iterations.
                comgra.my_recorder.register_tensor(
                    f"re_registered_chain", helper_partial_sums
                )
                comgra.my_recorder.register_tensor(
                    # This should refer to re_registered_chain if it exists, and otherwise (if commented out)
                    # to an automatically imported reference to helper_partial_sums from the previous iteration
                    f"reused_chain", helper_partial_sums * 1
                )
                comgra.my_recorder.register_tensor(
                    f"reused_copy", helper_copy2
                )
                # Test graph construction: split_test_in will be reached multiple times during backpropagation,
                # from the same reference last_encountered_reference, split_test_out
                split_test_in = helper_copy2 * 1
                combined = split_test_in + split_test_in * 2 + split_test_in * 3
                comgra.my_recorder.register_tensor(
                    f"split_test_in", split_test_in
                )
                split_test_out = combined * 1
                comgra.my_recorder.register_tensor(
                    f"split_test_out", split_test_out
                )
            if iteration == num_iterations - 1:
                comgra.my_recorder.register_tensor(f"target", target_tensor, is_target=True)
                # Calculate the loss and perform a backward pass as normal
                loss = self.criterion(output, target_tensor)
                self.optimizer.zero_grad()
                loss.backward()
                comgra.my_recorder.register_tensor(f"loss", loss, is_loss=True)
                accuracy = (output.argmax(dim=1).eq(target_tensor.argmax(dim=1))).float().mean()
                accuracy_first_ten = (output[:10, :].argmax(dim=1).eq(target_tensor[:10, :].argmax(dim=1))).float().mean()
                comgra.my_recorder.record_kpi_in_graph(
                    "loss", f"", loss,
                    timepoint=training_step,  # The timepoint uses the training_step by default
                )
                comgra.my_recorder.record_kpi_in_graph(
                    "accuracy", f"all", accuracy,
                )
                comgra.my_recorder.record_kpi_in_graph(
                    "accuracy", f"first_ten", accuracy_first_ten,
                )
                if update_the_model_parameters:
                    self.optimizer.step()
                comgra.my_recorder.record_current_gradients(f"gradients")
                if comgra.my_recorder.recording_is_active():
                    note = ""
                    if num_iterations == 1:
                        note = "-" * 20 + "\n"
                    note += (f"{self.current_configuration}, Step {training_step:5}: "
                             f"{num_iterations:2} iterations "
                             f"{'(training)' if update_the_model_parameters else ' (testing)'}  -  "
                             f"Loss = {loss.item():10.6f}  -  "
                             f"Accuracy on batch = {accuracy:2.3f}")
                    print(note)
                    comgra.my_recorder.add_note(note)
            comgra.my_recorder.finish_iteration()
        comgra.my_recorder.finish_batch()

    def generate_task_data(self):
        # Generate some datasets for training (lengths 1 to 10) and for testing (lengths 15 and 20)
        task_data = [
            (number_of_iterations, number_of_iterations <= 10, DataLoader(
                ExampleDataset(device=self.device, dataset_size=10_000, number_of_iterations=number_of_iterations),
                batch_size=self.batch_size,
                shuffle=True,
            ))
            # for number_of_iterations in list(range(1, 11)) + [15, 20]
            for number_of_iterations in list(range(1, 3)) + [5]
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
        return x


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
    # Inspect runtimes
    comgra.utilities.print_total_runtimes()


if __name__ == '__main__':
    main()

