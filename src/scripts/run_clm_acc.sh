
accelerate ../trainers/run_clm_acc.py \
--model_name_or_path='EleutherAI/polyglot-ko-5.8b' \
--train_file='/workspace/Coding/lm-trainer/src/KoAlpaca_v1.1a_textonly.json' \
--per_device_train_batch_size=1 \
--gradient_accumulation_steps=64 \