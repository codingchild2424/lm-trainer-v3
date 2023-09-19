
torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_dpo.py \
--model_name_or_path='gpt2' \
--train_file='Anthropic/hh-rlhf' \
--max_steps=1000 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=64 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot_dpo-v1' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--bf16 True \