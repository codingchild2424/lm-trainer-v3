
torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_rm.py \
--model_name_or_path='facebook/opt-350m' \
--train_file='Anthropic/hh-rlhf' \
--num_train_epochs=1 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot_rm-v1' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--use_peft=False \
--save_total_limit=1 \
--seq_length 512 \
--bf16 True \