#!/bin/bash
            
data_path="data/globem/GLOBEM_anxiety_train_all.json"
output_path="../Log/log/medalpaca-7b-batch12-epoch100-globem"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=2024 medalpaca/train.py \
    --model "medalpaca/medalpaca-7b" \
    --data_path "$data_path" \
    --output_dir "$output_path" \
    --train_in_8bit False \
    --use_lora True \
    --bf16 True \
    --tf32 True \
    --fp16 False \
    --gradient_checkpointing False \
    --global_batch_size 12 \
    --per_device_batch_size 12 \
    --num_epochs 100
