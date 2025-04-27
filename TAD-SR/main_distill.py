# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import argparse
from omegaconf import OmegaConf
from trainer import TrainerDistillDifIR as TrainerDistill

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
            "--save_dir",
            type=str,
            default="./saved_logs",
            help="Folder to save the checkpoints and training log",
            )
    parser.add_argument(
            "--resume",
            type=str,
            const=True,
            default="",
            nargs="?",
            help="Resume from the save_dir or checkpoint",
            )
    parser.add_argument(
            "--cfg_path",
            type=str,
            default="./configs/TAD-SR.yaml",
            help="Configs of yaml file",
            )
    parser.add_argument(
            "--steps",
            type=int,
            default=15,
            help="Hyper-parameters of diffusion steps",
            )
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = get_parser()

    configs = OmegaConf.load(args.cfg_path)
    #configs.diffusion.params.steps = args.steps

    # merge args to config
    for key in vars(args):
        if key in ['cfg_path', 'save_dir', 'resume']:
            configs[key] = getattr(args, key)

    trainer = TrainerDistill(configs)
    trainer.train()
