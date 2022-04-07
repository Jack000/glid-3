# GLID-3

GLID-3 is a combination of OpenAI GLIDE, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) and CLIP. The code is a modified version of [guided diffusion](https://github.com/openai/guided-diffusion)

main ideas:
- use same text conditioning as GLIDE
- instead of training a new text transformer, use the existing one from OpenAI CLIP
- instead of upsampling, do diffusion in the latent diffusion space
- add classifier-free guidance

[try it on colab](https://colab.research.google.com/drive/1x4p2PokZ3XznBn35Q5BBD6K6Zs-tot5t?usp=sharing)

# Install

You will need to install [latent diffusion](https://github.com/CompVis/latent-diffusion)
```
# then
git clone https://github.com/Jack000/glid-3
cd glid-3
pip install -e .
```

# Sampling from pre-trained models
note: this model is work in progress! image quality may not be ideal but it's usable
note2: this model is trained on photographic images only, and may not work well for illustrations and artwork

```
# first download latent diffusion model from CompVis (vq-f8)
wget https://ommer-lab.com/files/latent-diffusion/vq-f8.zip && unzip vq-f8.zip -d vq-f8

# download latest pretrained glid-3 model
wget https://dall-3.com/models/glid-3/ema-latest.pt

# default (fast PLMS sampling)
python sample.py --model_path ema-latest.pt --ldm_path vq-f8/model.ckpt --ldm_config_path vq-f8/config.yaml --width 256 --height 256 --batch_size 6 --num_batches 6 --text "a cyberpunk girl with a scifi neuralink device on her head"

# slow full ddpm run
python sample.py --ddpm --model_path ema-latest.pt --ldm_path vq-f8/model.ckpt --ldm_config_path vq-f8/config.yaml --width 256 --height 256 --batch_size 6 --num_batches 6 --text "a cyberpunk girl with a scifi neuralink device on her head"

# generated images saved to ./output/
```

# Training

Train with same flags as guided diffusion. Data directory should contain image and text files with the same name (image1.png image1.txt)

```
# on a single rtx 3090
MODEL_FLAGS="--ema_rate 0.9999 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 32 --learn_sigma True --noise_schedule cosine --num_channels 320 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 1e-5 --batch_size 12 --microbatch 4 --log_interval 1 --save_interval 5000 --vq_conf vq-f8/config.yaml --vq_model vq-f8/model.ckpt"
export OPENAI_LOGDIR=./logs/
python scripts/image_train_latent.py --data_dir /path/to/data $MODEL_FLAGS $TRAIN_FLAGS
```

# Fine tuning
```
mkdir logs

# download latest model, ema and optimizer from https://dall-3.com/models/glid-3/
# and put into your ./logs/

MODEL_FLAGS="--ema_rate 0.9999 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 32 --learn_sigma True --noise_schedule cosine --num_channels 320 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--resume_checkpoint logs/model6000000.pt --lr 1e-5 --batch_size 12 --microbatch 4 --log_interval 1 --save_interval 5000 --vq_conf vq-f8/config.yaml --vq_model vq-f8/model.ckpt"
export OPENAI_LOGDIR=./logs/
python scripts/image_train_latent.py --data_dir /path/to/data $MODEL_FLAGS $TRAIN_FLAGS
```

