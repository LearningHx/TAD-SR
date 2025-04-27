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

import os, sys
import argparse
from pathlib import Path

from omegaconf import OmegaConf
from sampler import Sampler

from utils.util_opts import str2bool
from basicsr.utils.download_util import load_file_from_url


_LINK = {
    'vqgan': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/autoencoder_vq_f4.pth',
    'vqgan_face256': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/celeba256_vq_f4_dim3_face.pth',
    'vqgan_face512': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/ffhq512_vq_f8_dim8_face.pth',
    'resshift_realsr': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_realsrx4_s15_v1.pth',
    'resshift_faceir': 'https://github.com/zsyOAOA/ResShift/releases/download/v2.0/resshift_faceir_s4.pth',
         }

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    parser.add_argument("-o", "--out_path", type=str, default="./results", help="Output path.")
    parser.add_argument("-r", "--ref_path", type=str, default=None, help="reference image")
    parser.add_argument("-s", "--steps", type=int, default=15, help="Diffusion length. (The number of steps that the model trained on.)")
    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("-is", "--infer_steps", type=int, default=None, help="Diffusion length for inference")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed.")
    parser.add_argument("--one_step", action="store_true")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--model_version", type=str, default=None)
    parser.add_argument(
            "--chop_size",
            type=int,
            default=512,
            choices=[512, 256],
            help="Chopping forward.",
            )
    parser.add_argument(
            "--task",
            type=str,
            default="realsrx4",
            choices=['realsrx4', 'bicsrx4_opencv', 'bicsrx4_matlab','faceir'],
            help="Chopping forward.",
            )
    parser.add_argument("--ddim", action="store_true")
    
    args = parser.parse_args()
    if args.infer_steps is None:
        args.infer_steps = args.steps
    print(f"[INFO] Using the inference step: {args.steps}")
    return args

def get_configs(args):
    ckpt_dir = Path('./weights')
    if not ckpt_dir.exists():
        ckpt_dir.mkdir()
    if args.config is None:
        if args.task == 'realsrx4':
            configs = OmegaConf.load('./configs/TAD-SR.yaml')
            vqgan_url = _LINK['vqgan']
            vqgan_path = ckpt_dir / f'autoencoder_vq_f4.pth'
            if args.ckpt is None:
                if args.model_version=='resshift_realsr':
                    ckpt_url = _LINK[args.model_version]
                    ckpt_path = ckpt_dir / f'resshift_realsrx4_s15_v1.pth'
                else:
                    ckpt_url = _LINK[args.model_version]
                    ckpt_path = ckpt_dir / f'TAD-SR.pth'
            else:
                ckpt_path = Path(args.ckpt)
        elif args.task == 'faceir':
            configs =  OmegaConf.load('./configs/TAD-faceir.yaml')
            vqgan_url = _LINK['vqgan_face512']
            vqgan_path = ckpt_dir / f'ffhq512_vq_f8_dim8_face.pth'
            if args.ckpt is None:
                if args.model_version=='resshift_faceir':
                    ckpt_url = _LINK[args.model_version]
                    ckpt_path = ckpt_dir / f'resshift_faceir_s4.pth'
                else:
                    ckpt_url = _LINK[args.model_version]
                    ckpt_path = ckpt_dir / f'TAD-faceir.pth'
            else:
                ckpt_path = Path(args.ckpt)
        else:
            raise TypeError(f"Unexpected task type: {args.task}!")
    else:
        configs = OmegaConf.load(args.config)
    # prepare the checkpoint
    if not ckpt_path.exists():
         load_file_from_url(
            url=ckpt_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=ckpt_path.name,
            )
    if not vqgan_path.exists():
         load_file_from_url(
            url=vqgan_url,
            model_dir=ckpt_dir,
            progress=True,
            file_name=vqgan_path.name,
            )
    print(f"[INFO] Using the checkpoint {ckpt_path}")

    configs.model.ckpt_path = str(ckpt_path)
    configs.diffusion.params.timestep_respacing = args.infer_steps
    configs.diffusion.params.sf = args.scale
    configs.autoencoder.ckpt_path = str(vqgan_path)

    # save folder
    if not Path(args.out_path).exists():
        Path(args.out_path).mkdir(parents=True)

    if args.chop_size == 512:
        chop_stride = 448
    elif args.chop_size == 256:
        chop_stride = 224
    else:
        raise ValueError("Chop size must be in [512, 384, 256]")

    return configs, chop_stride

def main():
    args = get_parser()

    configs, chop_stride = get_configs(args)

    resshift_sampler = Sampler(
            configs,
            chop_size=args.chop_size,
            chop_stride=chop_stride,
            chop_bs=1,
            use_fp16=True,
            seed=args.seed,
            ddim=args.ddim
            )

    resshift_sampler.inference(args.in_path, args.out_path, bs=1, noise_repeat=False, one_step=args.one_step)
    import evaluate
    evaluate.evaluate(args.out_path, args.ref_path, None)
    
    
if __name__ == '__main__':
    main()
