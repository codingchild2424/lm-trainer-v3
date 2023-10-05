# Works on A100 80G x8

####################################
# sft
# 4bit quantization for falcon-180B
###################################\

#!/bin/bash

epoch_nums=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)

for epoch_num in "${epoch_nums[@]}"
do
    torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_sft.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --train_file /workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko_instruction.jsonl \
    --num_train_epochs "$epoch_num" \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 64 \
    --torch_dtype float32 \
    --output_dir /workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-title-instruction-epoch_"$epoch_num" \
    --deepspeed ../ds_configs/ds_zero3-nooffload.json \
    --use_peft False \
    --save_total_limit 1 \
    --seq_length 512 \
    --bf16 True \
    --response_template '### 제목:'
done


for epoch_num in "${epoch_nums[@]}"
do
    torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_sft.py \
    --model_name_or_path EleutherAI/polyglot-ko-5.8b \
    --train_file /workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.jsonl \
    --num_train_epochs "$epoch_num" \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --torch_dtype float32 \
    --output_dir /workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-title-epoch_"$epoch_num" \
    --deepspeed ../ds_configs/ds_zero3-nooffload.json \
    --use_peft False \
    --save_total_limit 1 \
    --seq_length 512 \
    --bf16 True \
    --response_template '### 제목:'
done

####################################
# sft
# 4bit quantization for falcon-180B
####################################
# torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_sft.py \
# --model_name_or_path='tiiuae/falcon-180B' \
# --train_file='/workspace/Coding/lm-trainer/src/KoAlpaca_v1.1a_textonly.jsonl' \
# --num_train_epochs=1 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=64 \
# --output_dir='/workspace/Coding/lm-trainer/model_records/koalpaca_sft-v1' \
# --deepspeed=../ds_configs/ds_zero3-nooffload.json \
# --use_peft=False \
# --save_total_limit=1 \
# --seq_length 512 \
# --bf16 True \
# --response_template '### 답변:' \
# --load_in_4bit True \
# --use_peft True \
# --qlora True \
# --use_auth_token True \