"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_text_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import torch
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config

import random

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    from clip_custom import clip # make clip end up on the right device

    logger.log("loading LDM...")

    config = OmegaConf.load(args.vq_conf)
    pl_sd = torch.load(args.vq_model, map_location="cpu")
    sd = pl_sd["state_dict"]
    encoder = instantiate_from_config(config.model)
    encoder.load_state_dict(sd, strict=False)
    encoder.to(dist_util.dev())
    encoder.eval()
    set_requires_grad(encoder, False)

    del encoder.decoder
    del encoder.post_quant_conv
    del encoder.loss

    logger.log("loading clip...")
    clip_model, _ = clip.load('ViT-L/14', device=dist_util.dev(), jit=False)
    clip_model.eval().requires_grad_(False)
    set_requires_grad(clip_model, False)

    del clip_model.visual
    torch.cuda.empty_cache()
    print('using memory: ', torch.cuda.memory_allocated(0))
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())

    logger.log('total base parameters', sum(x.numel() for x in model.parameters()))

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_latent_data(
        encoder,
        clip_model,
        clip,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()

def load_latent_data(encoder, clip_model, clip, data_dir, batch_size, image_size):
    data = load_data(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=256,
        class_cond=False,
    )
    for batch, model_kwargs, text in data:
        text = clip.tokenize(text, truncate=True).to(dist_util.dev())

        text_emb, text_out = clip_model.encode_text(text, out=True)
        text_out = text_out.permute(0, 2, 1)

        text_blank = clip.tokenize(['']*batch.shape[0]).to(dist_util.dev())
        #text_blank = torch.zeros(batch.shape[0], 77, dtype=torch.int32).to(dist_util.dev())
        text_emb_blank, text_out_blank = clip_model.encode_text(text_blank, out=True)
        text_out_blank = text_out_blank.permute(0, 2, 1)


        for i in range(batch.shape[0]):
            if random.randint(0,100) < 20:
                text_emb[i] = text_emb_blank[i]
                text_out[i] = text_out_blank[i]

        model_kwargs["xf_proj"] = text_emb
        model_kwargs["xf_out"] = text_out

        batch = batch.to(dist_util.dev())
        emb, _, _ = encoder.encode(batch)

        yield emb, model_kwargs

def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        encoder_channels=None,
        vq_conf=None,
        vq_model=None,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults['encoder_channels'] = 768
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
