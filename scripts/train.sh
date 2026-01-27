#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 run.py \
    --workers 4 \
    --gpu '1' \
    --batch_size 2 \
    --test_batch_size 2 \
    --epoch 100 \
    --lr 1e-4 \
    --weight_decay 0.01 \
    --model "swin_multitask" \
    --dataset "downstream_vessel" \
    --dataset_name "DRIVE" \
    --csv_base_dir "/mnt/hdd/sdd/wangzh/datasets/" \
    --train_csv_name "train.csv" \
    --size 640 \
    --out_channels "(2,3,2,4,6)" \
    --class_weights "(1, 1, 1, 1, 1)" \
    --patch_size 4 \
    --window_size 10 \
    --depths "(2, 2, 18, 2)" \
    --num_heads "(4, 8, 16, 32)" \
    --feature_size 128 \
    --log_name "downstream_vessel_finetune_DRIVE_640" \
    --save_name "/mnt/hdd/sdd/wangzh/experiments/retsam/downstream_vessel_finetune/DRIVE_640/" \
    --seed 0 \
    --pretrained "/mnt/hdd/sdd/wangzh/checkpoints/retsam/retsam_v1_for_public.ckpt" \
    --description "retsam downstream vessel segmentation, DRIVE" \
    --save_results \
    --debug