# Works on A100 80G x8

####################################
# sft (with fp16)
# if you want to train polyglot-ko-12.8b, you must use fp16
####################################
# torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_sft.py \
# --model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
# --train_file='/workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/ise_book_data_1000.jsonl' \
# --num_train_epochs=1 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=64 \
# --torch_dtype=bfloat16 \
# --output_dir='/workspace/Coding/lm-trainer/model_records/polyglot_5.8b_sft_ise_book_data_1000_epoch1' \
# --deepspeed=../ds_configs/ds_zero3-nooffload.json \
# --use_peft=False \
# --save_total_limit=1 \
# --seq_length 512 \
# --bf16 True \
# --response_template '### 챗봇:' \


#!/bin/bash

# epoch_nums="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"

# for epoch in $epoch_nums;
# do

torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_sft.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='/workspace/home/uglee/Coding/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko_instruction.jsonl' \
--num_train_epochs=5 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--output_dir="/workspace/home/uglee/Coding/Coding/lm-trainer/model_records/polyglot_5.8b_sft_title_generation" \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--use_peft=False \
--save_total_limit=1 \
--seq_length 2048 \
--response_template '### 제목:'
