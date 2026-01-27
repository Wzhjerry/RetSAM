#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --workers 12 \
    --gpu '1' \
    --batch_size 4 \
    --test_batch_size 2 \
    --epoch 100 \
    --lr 1e-5 \
    --weight_decay 0.01 \
    --model "swin_multitask" \
    --dataset "maples_dr" \
    --size 640 \
    --out_channels "(2,3,2,4,6)" \
    --class_weights "(1, 1, 1.5)" \
    --patch_size 4 \
    --window_size 10 \
    --depths "(2, 2, 18, 2)" \
    --num_heads "(4, 8, 16, 32)" \
    --feature_size 128 \
    --log_name "swinunetr_base_multitask_finetune_maples_dr_640" \
    --save_name "/workspace/wangzhonghua/experiments/retsam/multitask_finetune/swinunetr_base/maples_dr_640/" \
    --seed 0 \
    --pretrained "/root/checkpoints/combined_8task_model.ckpt" \
    --description "retsam multitask segmentation, swinunetr with swin_b parameter setting, imagenet pretrain, maples_dr" \
    --save_results \
    --debug