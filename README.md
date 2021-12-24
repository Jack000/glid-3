# GLID-3

GLID-3 is a combination of OpenAI GLIDE, [Latent Diffusion](https://github.com/CompVis/latent-diffusion) and CLIP

main ideas:
- use same text conditioning as GLIDE
- instead of training a new text transformer, use the existing one from OpenAI CLIP
- instead of upsampling, do diffusion in the latent diffusion space
- add classifier-free guidance

# Sampling from pre-trained models

(work in progress, no pretrained models yet)

```
# first download model from latent diffusion (vq-f8-n256)
sample.py --model_path path/to/model.pt --ldm_path vq-f8-n256/model.ckpt --ldm_config_path vq-f8-n256/config.yaml --text "a cute puppy"
```

# Training

Train with same flags as guided diffusion. Data directory should contain image and text files with the same name (image1.png image1.txt)

```
MODEL_FLAGS="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 32 --learn_sigma True --noise_schedule linear --num_channels 320 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
TRAIN_FLAGS="--lr 6e-5 --batch_size 48 --microbatch 8 --log_interval 1 --save_interval 5000 --vq_conf vq-f8-n256/config.yaml --vq_model vq-f8-n256/model.ckpt"
export OPENAI_LOGDIR=./logs/
mpiexec -n 4 python scripts/image_train_latent.py --data_dir ./data $MODEL_FLAGS $TRAIN_FLAGS
```
