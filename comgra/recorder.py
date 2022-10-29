from pathlib import Path

import torch


class ComgraRecorder:

    def __init__(self, comgra_root_path, group, trial_id):
        comgra_root_path = Path(comgra_root_path)
        assert comgra_root_path.exists()
        self.trial_id = trial_id
        self.group_path = comgra_root_path / group
        self.trial_path = self.group_path / trial_id
        print(self.trial_path)
        self.trial_path.mkdir(parents=True, exist_ok=True)

    def track_module(self, module_name, module: torch.Module):
        pass

    def register_tensor(self, tensor_name, tensor: torch.Tensor):
        pass

    def start_forward_pass(self):
        pass

    def start_backward_pass(self):
        pass  # TODO this should be optional since there won't always be a loss.

    def finish_iteration(self):
        pass
