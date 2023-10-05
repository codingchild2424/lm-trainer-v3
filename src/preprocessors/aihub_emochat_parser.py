import pandas as pd
import os
import json
from tqdm import tqdm


def save_to_json_file(json_response, dst_path: str):
    with open(dst_path, 'a') as outfile:
        json.dump(json_response, outfile, ensure_ascii=False)
        outfile.write('\n')


def main():
    train_data_path = "/workspace/datasets/NLP/raw_datasets/ai_hub/018.감성대화/Training_221115_add/라벨링데이터/감성대화말뭉치(최종데이터)_Training.json"
    valid_data_path = "/workspace/datasets/NLP/raw_datasets/ai_hub/018.감성대화/Validation_221115_add/라벨링데이터/감성대화말뭉치(최종데이터)_Validation.json"

    dst_path = "/workspace/Coding/lm-trainer/datasets/pre_datasets/cl_data/aihub_emochat.json"

    file_path_list = [train_data_path, valid_data_path]

    with tqdm(file_path_list) as pbar:
        pbar.set_description("Processing")

        for file_path in pbar:
            with open(file_path, 'r', encoding='utf-8') as f:
                # json file
                json_data = json.load(f)
                
                for json_text in json_data:
                    
                    dialogues = []
                    
                    for key in list(json_text['talk']['content'].keys()):
                        if key.startswith("HS"):
                            dialogues.append("Human:" + json_text['talk']['content'][key])
                        elif key.startswith("SS"):
                            dialogues.append("System:" + json_text['talk']['content'][key])
                            
                    full_text = "\n".join(dialogues)
                    
                    json_full_text = {'text': full_text + "<|endoftext|>"}
                            
                    save_to_json_file(json_full_text, dst_path=dst_path)


if __name__ == '__main__':
    main()