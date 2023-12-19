#!/bin/bash

epoch_nums=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for epoch_num in "${epoch_nums[@]}"
do
    torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_new.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --train_file /workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.json \
    --num_train_epochs "$epoch_num" \
    --block_size 1024 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --torch_dtype bfloat16 \
    --fp16 \
    --output_dir /workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-tinystories_ko_normal_epoch_"$epoch_num" \
    --deepspeed ../ds_configs/ds_zero3-nooffload.json \
    --do_train \
    --save_strategy epoch \
    --logging_strategy steps \
    --logging_first_step \
    --save_total_limit 1 \
    --run_name polyglot-ko-5.8b-tinystories_ko_epoch_"$epoch_num"
done
