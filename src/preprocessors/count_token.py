from transformers import AutoTokenizer
import json
from tqdm import tqdm

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

data_path = "/workspace/Coding/lm-trainer/datasets/pre_datasets/panyo_data/pangyo_corpus.json"

with open(data_path, "r") as f:
    json_data_list = []# read line by line
    for line in f:
        line_text = json.loads(line)['text']
        json_data_list.append(line_text)

token_count_num = 0

# Tokenize the text
for json_data in tqdm(json_data_list):
    tokens = tokenizer.tokenize(json_data)
    # Count the number of tokens
    token_count = len(tokens)
    token_count_num += token_count

print("Number of tokens:", token_count_num) # 703 / 2543545
