
torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_ppo.py \
--model_name_or_path='/workspace/Coding/lm-trainer/model_records/polyglot_5.8b_sft_ise_book_data_1000_epoch10' \
--train_file='/workspace/Coding/lm-trainer/datasets/pre_datasets/ppo_data/ise_book_data_1000.json' \
--num_train_epochs=10 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--output_dir='/workspace/Coding/lm-trainer/model_records/polyglot_ppo-v1' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--use_peft=False \
--save_total_limit=1 \
--seq_length 512 \
--bf16 True \
--reward_model_name_or_path='/workspace/Coding/lm-trainer/model_records/hate_speech_koelectra_v2' \