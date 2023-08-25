# for debug
# https://stackoverflow.com/questions/69693950/error-some-nccl-operations-have-failed-or-timed-out

export NCCL_DEBUG=INFO

# #!/bin/bash

# accelerate config
#
# In which compute environment are you running?
# This machine
# ----------------------------------------------------------------------------------------------------------------------------------------Which type of machine are you using?
# multi-GPU
# How many different machines will you use (use more than 1 for multi-node training)? [1]: 1
# Do you wish to optimize your script with torch dynamo?[yes/NO]:no
# Do you want to use DeepSpeed? [yes/NO]: yes
# Do you want to specify a json file to a DeepSpeed config? [yes/NO]: no
# ----------------------------------------------------------------------------------------------------------------------------------------What should be your DeepSpeed's ZeRO optimization stage?
# 3
# ----------------------------------------------------------------------------------------------------------------------------------------Where to offload optimizer states?
# cpu
# ----------------------------------------------------------------------------------------------------------------------------------------Where to offload parameters?
# cpu
# How many gradient accumulation steps you're passing in your script? [1]: 1
# Do you want to use gradient clipping? [yes/NO]: no
# Do you want to save 16-bit model weights when using ZeRO Stage-3? [yes/NO]: no
# Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]: no
# How many GPU(s) should be used for distributed training? [1]:8
# ----------------------------------------------------------------------------------------------------------------------------------------Do you wish to use FP16 or BF16 (mixed precision)?
# fp16 no

# epochs=(1 2 3 4 5 6 7 8 9 10)

accelerate launch run_clm_no_trainer.py \
    --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
    --train_file='../datasets/pre_datasets/KoAlpaca_v1.1a_textonly.json' \
    --num_train_epochs=1 \
    --per_device_train_batch_size=2 \
    --gradient_accumulation_steps=32 \
    --output_dir="model_records/polyglot-test-epochs-1-v3"

# accelerate launch run_clm_no_trainer.py \
#     --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
#     --train_file='../datasets/pre_datasets/kowiki.json' \
#     --num_train_epochs=1 \
#     --per_device_train_batch_size=3 \
#     --gradient_accumulation_steps=128 \
#     --output_dir="model_records/polyglot-plus-kowiki-epochs-1"
#     --use_rope_scaling=False

# accelerate launch run_clm_no_trainer.py \
#     --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
#     --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
#     --num_train_epochs=2 \
#     --per_device_train_batch_size=3 \
#     --gradient_accumulation_steps=128 \
#     --output_dir="model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-2" \
#     --use_rope_scaling=False

# accelerate launch run_clm_no_trainer.py \
#     --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
#     --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
#     --num_train_epochs=3 \
#     --per_device_train_batch_size=3 \
#     --gradient_accumulation_steps=128 \
#     --output_dir="model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-3" \
#     --use_rope_scaling=False

# accelerate launch run_clm_no_trainer.py \
#     --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
#     --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
#     --num_train_epochs=4 \
#     --per_device_train_batch_size=3 \
#     --gradient_accumulation_steps=128 \
#     --output_dir="model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-4" \
#     --use_rope_scaling=False

# accelerate launch run_clm_no_trainer.py \
#     --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
#     --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
#     --num_train_epochs=5 \
#     --per_device_train_batch_size=3 \
#     --gradient_accumulation_steps=128 \
#     --output_dir="model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-5" \
#     --use_rope_scaling=False

# for i in "${epochs[@]}";
# do
#     accelerate launch run_clm_no_trainer.py \
#     --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
#     --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
#     --num_train_epochs=$i \
#     --per_device_train_batch_size=3 \
#     --gradient_accumulation_steps=128 \
#     --output_dir="model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-$i" \
#     --use_rope_scaling=False
# done

# after accelerate config, launch this command
# accelerate launch run_clm_no_trainer.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
# --num_train_epochs=6 \
# --block_size=1024 \
# --per_device_train_batch_size=3 \
# --gradient_accumulation_steps=128 \
# --output_dir='model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-6' \
# --use_rope_scaling False \
#--checkpointing_steps='epoch' \

# accelerate launch run_clm_no_trainer.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
# --num_train_epochs=7 \
# --block_size=1024 \
# --per_device_train_batch_size=3 \
# --gradient_accumulation_steps=128 \
# --output_dir='model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-7' \
# --use_rope_scaling False \

# accelerate launch run_clm_no_trainer.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
# --num_train_epochs=8 \
# --block_size=1024 \
# --per_device_train_batch_size=3 \
# --gradient_accumulation_steps=128 \
# --output_dir='model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-8' \
# --use_rope_scaling False \

# accelerate launch run_clm_no_trainer.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
# --num_train_epochs=9 \
# --block_size=1024 \
# --per_device_train_batch_size=3 \
# --gradient_accumulation_steps=128 \
# --output_dir='model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-9' \
# --use_rope_scaling False \

# accelerate launch run_clm_no_trainer.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction-20000.json' \
# --num_train_epochs=10 \
# --block_size=1024 \
# --per_device_train_batch_size=3 \
# --gradient_accumulation_steps=128 \
# --output_dir='model_records/ko_tinystories-v2-gpt4-instruction-20000-epochs-10' \
# --use_rope_scaling False \

#--resume_from_checkpoint model_records/KoOrca-v1-epochs-1/step_0 \

# accelerate launch run_clm_no_trainer.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='../datasets/KoAlpaca_v1.1a_textonly.json' \
# --num_train_epochs=1 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=128 \
# --output_dir='model_records/KoAlpaca'


