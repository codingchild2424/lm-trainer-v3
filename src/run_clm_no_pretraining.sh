# for debug
# https://stackoverflow.com/questions/69693950/error-some-nccl-operations-have-failed-or-timed-out

export NCCL_DEBUG=INFO

# #!/bin/bash

# epochs=(1 2 3 4 5 6 7 8 9 10)

accelerate launch run_clm_no_pretraining.py \
    --config_name='EleutherAI/polyglot-ko-1.3b'\
    --tokenizer_name='EleutherAI/polyglot-ko-1.3b'\
    --train_file='../datasets/pre_datasets/KoAlpaca_v1.1a_textonly.json' \
    --num_train_epochs=1 \
    --per_device_train_batch_size=3 \
    --gradient_accumulation_steps=128 \
    --output_dir="model_records/polyglot-1.3b-pretraining"

# accelerate launch run_clm_no_trainer.py \
#     --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
#     --train_file='../datasets/pre_datasets/kowiki.json' \
#     --num_train_epochs=1 \
#     --per_device_train_batch_size=3 \
#     --gradient_accumulation_steps=128 \
#     --output_dir="model_records/polyglot-plus-kowiki-epochs-1"
    #--use_rope_scaling=False

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


