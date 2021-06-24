import argparse
from pathlib import Path
import os

import ray
from ray import tune
from ray.rllib.examples.models.custom_loss_model import CustomLossModel, TorchCustomLossModel
from custom_loss_l2_model_lstm import TorchCustomLossL2ModelLSTM
from ray.rllib.models import ModelCatalog
from ray.rllib.utils.framework import try_import_tf

tf1, tf, tfv = try_import_tf()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "tfe", "torch"],
    default="torch",
    help="The DL framework specifier.")
parser.add_argument("--stop-iters", type=int, default=10)

if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()

    ModelCatalog.register_custom_model("custom_loss", TorchCustomLossL2ModelLSTM)

    config = {
        "env": "CartPole-v0",
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        "num_workers": 0,
        "model": {
            "custom_model": "custom_loss",
            "use_lstm": True,
            # "lstm_cell_size": 128,
            # "lstm_use_prev_action": True,
            # "lstm_use_prev_reward": True,
        },

        "framework": args.framework,
    }

    stop = {
        "training_iteration": args.stop_iters,
    }

    tune.run("PPO", config=config, stop=stop, verbose=1)
