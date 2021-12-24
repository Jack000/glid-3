import gc
import io
import math
import sys

from PIL import Image
import requests
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import functional as TF
from tqdm.notebook import tqdm

import numpy as np

import clip
from guided_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults

from dalle_pytorch import DiscreteVAE, VQGanVAE

from einops import rearrange
from math import log2, sqrt

import argparse

from clip_custom import clip
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

# argument parsing

parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str, default = 'model015000.pt',
                   help='path to the diffusion model')

parser.add_argument('--ldm_path', type=str, default = './vq-f8-n256/model.ckpt',
                   help='path to the LDM first stage model')

parser.add_argument('--ldm_config_path', type=str, default = './vq-f8-n256/config.yaml',
                   help='path to the LDM first stage config. This should be a .yaml file')

parser.add_argument('--text', type = str, required = False,
                    help='your text prompt, separate with | characters')

parser.add_argument('--num_batches', type = int, default = 1, required = False,
                    help='number of batches')

parser.add_argument('--batch_size', type = int, default = 1, required = False,
                    help='batch size')

parser.add_argument('--output_width', type = int, default = 256, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--output_height', type = int, default = 256, required = False,
                    help='image size of output (multiple of 8)')

parser.add_argument('--seed', type = int, default=0, required = False,
                    help='random seed')

parser.add_argument('--cpu', dest='cpu', action='store_true')

args = parser.parse_args()

def fetch(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        r = requests.get(url_or_path)
        r.raise_for_status()
        fd = io.BytesIO()
        fd.write(r.content)
        fd.seek(0)
        return fd
    return open(url_or_path, 'rb')

device = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
print('Using device:', device)

model_params = {
    'attention_resolutions': '32, 16, 8',
    'class_cond': False,
    'diffusion_steps': 1000,
    'rescale_timesteps': True,
    'timestep_respacing': '1000',  # Modify this value to decrease the number of
                                   # timesteps.
    'image_size': 32,
    'learn_sigma': True,
    'noise_schedule': 'linear',
    'num_channels': 320,
    'num_head_channels': 64,
    'num_res_blocks': 3,
    'encoder_channels': 512,
    'resblock_updown': True,
    'use_fp16': True,
    'use_scale_shift_norm': True
}

model_config = model_and_diffusion_defaults()
model_config.update(model_params)

if args.cpu:
    model_config['use_fp16'] = False

# Load models
model, diffusion = create_model_and_diffusion(**model_config)
model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
model.requires_grad_(False).eval().to(device)

if model_config['use_fp16']:
    model.convert_to_fp16()
elif args.cpu:
    model.convert_to_fp32()

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

# LDM
config = OmegaConf.load(args.ldm_config_path)
pl_sd = torch.load(args.ldm_path, map_location="cpu")
sd = pl_sd["state_dict"]
ldm = instantiate_from_config(config.model)
ldm.load_state_dict(sd, strict=False)
ldm.to(device)
ldm.eval()
set_requires_grad(ldm, False)

# clip
clip_model, _ = clip.load('ViT-B/16', device=device, jit=False)
clip_model.eval().requires_grad_(False)
set_requires_grad(clip_model, False)
del clip_model.visual

def do_run():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    text = clip.tokenize([args.text]*args.batch_size, truncate=True).to(device)

    text_emb, text_out = clip_model.encode_text(text, out=True)
    text_out = text_out.permute(0, 2, 1)

    text_blank = clip.tokenize(['']*args.batch_size).to(device)

    text_emb_blank, text_out_blank = clip_model.encode_text(text_blank, out=True)
    text_out_blank = text_out_blank.permute(0, 2, 1)

    print(text_emb.shape, text_out.shape)

    kwargs = { "xf_proj": torch.cat([text_emb, text_emb_blank], dim=0), "xf_out": torch.cat([text_out, text_out_blank], dim=0) }

    guidance_scale = 1.0

    # Create a classifier-free guidance sampling function
    def model_fn(x_t, ts, **kwargs):
        half = x_t[: len(x_t) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = model(combined, ts, **kwargs)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    cur_t = None

    if model_config['timestep_respacing'].startswith('ddim'):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    for i in range(args.num_batches):
        cur_t = diffusion.num_timesteps - 1

        samples = sample_fn(
            model_fn,
            (args.batch_size*2, 4, int(args.output_height/8), int(args.output_width/8)),
            clip_denoised=False,
            model_kwargs=kwargs,
            cond_fn=None,
            device=device,
            progress=True,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            if j % 50 == 0 or cur_t == -1 or j == 999:
                for k, image in enumerate(sample['pred_xstart'][:args.batch_size]):
                    print(image.min(), image.max())
                    out = ldm.decode(image.unsqueeze(0))

                    out = TF.to_pil_image(out.squeeze(0).add(1).div(2).clamp(0, 1))

                    filename = f'progress_{i * args.batch_size + k:05}.png'
                    out.save(filename)

gc.collect()
do_run()