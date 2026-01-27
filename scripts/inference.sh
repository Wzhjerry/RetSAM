CUDA_VISIBLE_DEVICES=2 python3 inference.py \
    --input_dir "/mnt/hdd/sdd/wangzh/datasets/retsam/webshow" \
    --output_dir "/mnt/hdd/sdd/wangzh/datasets/retsam/webshow_results_public" \
    --model_path /mnt/hdd/sdd/wangzh/checkpoints/retsam/retsam_v1_for_public.ckpt \
    --multitask \
    --device cuda \
    --output_channels "(2,3,2,4,6)" \
    --has_coordinate_head \
    --num_coordinates 2 \
    --enable_analysis
