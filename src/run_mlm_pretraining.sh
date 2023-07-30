# Pretraining Code, Works on A100 80G x 8
# for pretraining, you have to use model_type instead of model_name_or_path
torchrun --nproc_per_node=8 --master_port=34321 run_mlm.py \
--model_type='deberta-v2' \
--tokenizer_name='' \
--train_file='../datasets/pre_datasets/beomi.json' \
--num_train_epochs=1 \
--per_device_train_batch_size=2 \
--gradient_accumulation_steps=64 \
--output_dir='model_records' \
--deepspeed=ds_zero3-nooffload.json \
--do_train \
--save_strategy='epoch' \
--logging_strategy='steps' \
--logging_first_step \
--save_total_limit=1 \
--run_name='deberta-v3-test' \
--overwrite_output_dir True \
#--block_size=1024 \
#--low_cpu_mem_usage True \
#--torch_dtype=float16 \
#--fp16 \
#--model_name_or_path='microsoft/deberta-v3-large' \

# self.embedding_size, config.vocab_size
