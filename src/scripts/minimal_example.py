import torch
import torch.nn as nn
import torch.optim as optim
import comgra
from comgra.objects import DecisionMakerForRecordingsFrequencyPerType
from comgra.recorder import ComgraRecorder

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(5, 5)

    def forward(self, x):
        return self.linear(x)

# Initialize comgra
comgra.my_recorder = ComgraRecorder(
    comgra_root_path="/my/path/for/storing/data",
    group="name_of_experiment_group",
    trial_id="example_trial",
    decision_maker_for_recordings=DecisionMakerForRecordingsFrequencyPerType(min_training_steps_difference=10),
)

# Create model, loss function, and optimizer
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
comgra.my_recorder.track_module("main_model", model)

# Generate some dummy data
inputs = torch.randn(100, 5)
targets = 2 * inputs + torch.randn(100, 5) * 0.1

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    comgra.my_recorder.start_batch(epoch, inputs.shape[0])
    comgra.my_recorder.start_iteration()
    # Forward
    comgra.my_recorder.register_tensor("inputs", inputs, is_input=True)
    outputs = model(inputs)
    comgra.my_recorder.register_tensor("outputs", outputs)
    comgra.my_recorder.register_tensor("targets", targets, is_target=True)
    loss = criterion(outputs, targets)
    comgra.my_recorder.register_tensor("loss", loss, is_loss=True)
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    comgra.my_recorder.record_current_gradients(f"gradients")
    comgra.my_recorder.finish_iteration()
    comgra.my_recorder.finish_batch()
comgra.my_recorder.finalize()

