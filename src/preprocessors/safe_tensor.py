import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-title-instruction-epoch_20"

model = AutoModelForCausalLM.from_pretrained(model_path)

# add special tokens


tokenizer = AutoTokenizer.from_pretrained(model_path)

# add special tokens <|endoftext|> to tokenizer
tokenizer.add_special_tokens(
    {"additional_special_tokens": ["<|endoftext|>"]}
)

model.resize_token_embeddings(len(tokenizer))

save_file(model, "model.safetensors")