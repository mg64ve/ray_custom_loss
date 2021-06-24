import numpy as np

from ray.rllib.models.modelv2 import ModelV2, restore_original_dimensions
from ray.rllib.models.tf.tf_action_dist import Categorical
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_action_dist import TorchCategorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.offline import JsonReader

torch, nn = try_import_torch()


class TorchCustomLossL2Model(TorchModelV2, nn.Module):
    """PyTorch version of the CustomLossModel above."""

    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name, input_files):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        nn.Module.__init__(self)

        self.hascustomloss=True

        self.input_files = input_files
        # Create a new input reader per worker.
        self.reader = JsonReader(self.input_files)
        self.fcnet = TorchFC(
            self.obs_space,
            self.action_space,
            num_outputs,
            model_config,
            name="fcnet")

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        # Delegate to our FCNet.
        return self.fcnet(input_dict, state, seq_lens)

    @override(ModelV2)
    def custom_loss(self, policy_loss,
                    loss_inputs):

        l2_lambda = 0.01

        l2_reg = torch.tensor(0.)
        for param in self.parameters():
            l2_reg += torch.norm(param)
        self.l2_loss = l2_lambda * l2_reg

        assert self.l2_loss.requires_grad, "l2 loss no gradient"

        custom_loss = self.l2_loss

        # depending on input add loss
        if self.hascustomloss: #in case you want to only regularize base on a config, ...
            if isinstance(policy_loss, list):
                return [single_loss+custom_loss for single_loss in policy_loss]
            else:
                return policy_loss+custom_loss

        return policy_loss

    def metrics(self):
        metrics = {
            "weight_loss": self.l2_loss.item(),
        }
       # you can print them to command line here. with Torch models its somehow not reportet to the logger
        print(metrics)
        return metrics
