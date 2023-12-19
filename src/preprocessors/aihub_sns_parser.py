import pandas as pd
import os
import json
from tqdm import tqdm


def save_to_json_file(json_response, dst_path: str):
    with open(dst_path, 'a') as outfile:
        json.dump(json_response, outfile, ensure_ascii=False)
        outfile.write('\n')


def main():
    train_data_folder_path = "/workspace/datasets/NLP/raw_datasets/ai_hub/한국어 SNS/Training/[라벨]한국어SNS_train"
    valid_data_folder_path = "/workspace/datasets/NLP/raw_datasets/ai_hub/한국어 SNS/Validation/[라벨]한국어SNS_valid"

    dst_path = "/workspace/Coding/lm-trainer/datasets/pre_datasets/cl_data/aihub_sns.json"

    train_data_path_list = [os.path.join(train_data_folder_path, i) for i in os.listdir(train_data_folder_path)]
    valid_data_path_list = [os.path.join(valid_data_folder_path, i) for i in os.listdir(valid_data_folder_path)]

    file_path_list = train_data_path_list + valid_data_path_list

    with tqdm(file_path_list) as pbar:
        pbar.set_description("Processing")

        for file_path in pbar:
            with open(file_path, 'r', encoding='utf-8') as f:
                # json file
                json_data = json.load(f)
                
                for json_part in json_data['data']:
                    
                    if json_part['body'] == None:
                        continue
                    else:
                    
                        dialogues = []
                    
                        for json_part in json_part['body']:
                            dialouge = json_part["participantID"] + ":" + json_part["utterance"]
                            dialogues.append(dialouge)
                            
                        full_text = "\n".join(dialogues)
                        
                        json_full_text = {'text': full_text + "<|endoftext|>"}
                                
                        save_to_json_file(json_full_text, dst_path=dst_path)

if __name__ == '__main__':
    main()