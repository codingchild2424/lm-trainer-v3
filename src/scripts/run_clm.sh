# Works on A100 80G x8

####################################
# Continual Learning
# - For continual learinng, model_name_or_path must be changed
####################################
# torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='/workspace/Coding/lm-trainer/src/KoAlpaca_v1.1a_textonly.json' \
# --num_train_epochs=1 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=64 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='../model_results/polyglot-12.8b-kowiki20000-v2' \
# --deepspeed=../ds_configs/ds_zero3-nooffload.json \
# --do_train \
# --save_strategy='epoch' \
# --logging_strategy='steps' \
# --logging_first_step \
# --save_total_limit=1 \
# --run_name='polyglot-12.8b-kowiki20000-v2'

# meta-llama/Llama-2-7b-hf
# mistralai/Mistral-7B-v0.1

torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
<<<<<<< HEAD
--train_file='/workspace/home/uglee/Coding/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko_instruction.json' \
--num_train_epochs=5 \
--block_size=2048 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=32 \
--torch_dtype=float16 \
--output_dir='/workspace/home/uglee/Coding/Coding/lm-trainer/model_records/polyglot_5.8b_title_generation' \
=======
--train_file='/workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.json' \
--num_train_epochs=1 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--fp16 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-tinystories_ko_normal_epoch1' \
>>>>>>> 6032c2126b839a4706cd30a7a781fbec259a2e59
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
<<<<<<< HEAD
--run_name='polyglot_5.8b_title_generation'

# /workspace/datasets/NLP/pre_datasets/pre_for_clm/beomi.jsonl
# /workspace/Coding/lm-trainer/datasets/pre_datasets/panyo_data/pangyo_corpus.json


=======
--run_name='polyglot-ko-5.8b-tinystories_ko_epoch1'
>>>>>>> 6032c2126b839a4706cd30a7a781fbec259a2e59

torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='/workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.json' \
--num_train_epochs=2 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--fp16 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-tinystories_ko_normal_epoch2' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='polyglot-5.8b-total-385159-epoch2'

torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='/workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.json' \
--num_train_epochs=3 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--fp16 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-tinystories_ko_normal_epoch3' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='polyglot-5.8b-total-385159-epoch3'

torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='/workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.json' \
--num_train_epochs=4 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--fp16 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-tinystories_ko_normal_epoch4' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='polyglot-5.8b-total-385159-epoch4'

torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='/workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.json' \
--num_train_epochs=5 \
--block_size=1024 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--fp16 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-tinystories_ko_normal_epoch5' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='polyglot-5.8b-total-385159-epoch5'

# torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='/workspace/Coding/lm-trainer/src/KoAlpaca_v1.1a_textonly.json' \
# --num_train_epochs=1 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=64 \
# --torch_dtype=float16 \
# --fp16 \
# --output_dir='../model_results/polyglot-12.8b-kowiki20000-v2' \
# --deepspeed=../ds_configs/ds_zero3-nooffload.json \
# --do_train \
# --save_strategy='epoch' \
# --logging_strategy='steps' \
# --logging_first_step \
# --save_total_limit=1 \
# --run_name='polyglot-12.8b-kowiki20000-v2'