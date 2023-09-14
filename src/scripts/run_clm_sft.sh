# Works on A100 80G x8

####################################
# sft (with fp16)
# if you want to train polyglot-ko-12.8b, you must use fp16
####################################
torchrun --nproc_per_node=8 --master_port=34321 ../trainers/run_clm_sft.py \
--model_name_or_path='EleutherAI/polyglot-ko-12.8b' \
--train_file='/workspace/Coding/lm-trainer/src/KoAlpaca_v1.1a_textonly.jsonl' \
--num_train_epochs=1 \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=64 \
--torch_dtype=bfloat16 \
--output_dir='../../model_results/polyglot-12.8b-kowiki20000-v2' \
--deepspeed=../ds_configs/ds_zero3-nooffload.json \
--use_peft=False \
--save_total_limit=1 \
--seq_length 512 \
--bf16 True \
--fotmatting_prompts_func simple \
