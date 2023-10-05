
python gen_data_tinystories.py \
--src_path /workspace/Coding/lm-trainer/src/instruction_generators/seeds/tinystories_seed2.json \
--dst_path /workspace/Coding/lm-trainer/datasets/pre_datasets/inst_data/tinystories_ko.json \
--prompt_template_path /workspace/Coding/lm-trainer/src/instruction_generators/prompt_templates/tinystories.txt \
--model_name gpt-4 \
--preprocess_type prompt_chain_tinystories \