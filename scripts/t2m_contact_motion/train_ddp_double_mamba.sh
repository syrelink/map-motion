#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME="CMDM-DoubleMamba-H3D-CFG"
CONTACT_PATH="outputs/CDM-Perceiver-H3D"  # ADM 模型路径

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_ddp.py \
    task=text_to_motion_contact_motion_gen \
    model=cmdm_double_mamba \
    model.use_cfg=true \
    model.cfg_dropout_prob=0.1 \
    model.mamba_layers=2 \
    model.mamba_d_state=32 \
    model.mamba_d_conv=8 \
    exp_name=${MODEL_NAME} \
    diffusion.num_diffusion_timesteps=1000 \
    task.dataset.contact_dir=${CONTACT_PATH}/contact \
    training.batch_size=64 \
    training.num_epochs=300 \
    training.save_interval=10 \
    training.eval_interval=10 \
    training.use_ema=true \
    training.ema_decay=0.9999 \
    optimizer.type=adamw \
    optimizer.lr=1e-4 \
    optimizer.weight_decay=0.01 \
    scheduler.type=cosine \
    scheduler.warmup_epochs=10 \
    scheduler.max_epochs=300 \
    scheduler.min_lr=1e-6