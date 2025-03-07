#!/bin/bash
            
data_path="data/lifesnapes/LifeSnaps_stress_resilience_train_all.json"
output_path="log/medalpaca-7b-batch12-epoch1000-lifesnapes"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=2024 medalpaca/train_v2.py \
    --model "medalpaca/medalpaca-7b" \
    --data_path "$data_path" \
    --output_dir "$output_path" \
    --train_in_8bit False \
    --use_lora True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --gradient_checkpointing False \
    --global_batch_size 32 \
    --per_device_batch_size 32 \
    --num_epochs 1000
