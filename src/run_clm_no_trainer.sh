# for debug
# https://stackoverflow.com/questions/69693950/error-some-nccl-operations-have-failed-or-timed-out

export NCCL_DEBUG=INFO

# Before we start, use accelerate config
# The option

# after accelerate config, launch this command
accelerate launch run_clm_no_trainer.py \
--model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
--train_file='../datasets/pre_datasets/ko_tinystories-v2-gpt4-instruction.json' \
--num_train_epochs=3 \
--block_size=1024 \
--per_device_train_batch_size=3 \
--gradient_accumulation_steps=128 \
--output_dir='model_records/ko_tinystories-v2-gpt4-instruction-epochs-3-rope' \
--checkpointing_steps='epoch' \
#--resume_from_checkpoint model_records/KoOrca-v1-epochs-1/step_0 \

# accelerate launch run_clm_no_trainer.py \
# --model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
# --train_file='../datasets/KoAlpaca_v1.1a_textonly.json' \
# --num_train_epochs=1 \
# --block_size=1024 \
# --per_device_train_batch_size=1 \
# --gradient_accumulation_steps=128 \
# --output_dir='model_records/KoAlpaca'


