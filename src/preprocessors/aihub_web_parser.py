import pandas as pd
import os
import json
from tqdm import tqdm


def save_to_json_file(json_response, dst_path: str):
    with open(dst_path, 'a') as outfile:
        
        #print(json_response)
        
        json.dump(json_response, outfile, ensure_ascii=False)
        outfile.write('\n')
        
        #print("save json_response")


def main():
    train_folder_path = "/workspace/datasets/NLP/raw_datasets/ai_hub/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/1.Training/라벨링데이터"
    valid_folder_path = "/workspace/datasets/NLP/raw_datasets/ai_hub/030.웹데이터 기반 한국어 말뭉치 데이터/01.데이터/2.Validation/라벨링데이터"
    dst_path = "/workspace/Coding/lm-trainer/datasets/pre_datasets/cl_data/aihub_web.json"


    train_sub_folder_list = os.listdir(train_folder_path)
    train_sub_folder_name_list = [os.path.join(train_folder_path, x) for x in train_sub_folder_list]

    valid_sub_folder_list = os.listdir(valid_folder_path)
    valid_sub_folder_name_list = [os.path.join(valid_folder_path, x) for x in valid_sub_folder_list]

    # concat all paths
    all_folder_path_list = train_sub_folder_name_list + valid_sub_folder_name_list



    full_text_list = []

    with tqdm(all_folder_path_list) as pbar:
        pbar.set_description("Processing")
        for folder_path in pbar:
            file_list = os.listdir(folder_path)
            file_list = [os.path.join(folder_path, x) for x in file_list]
            #print("file_list: ", file_list)
            
            # one file
            for file_path in file_list:
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    # json file
                    json_data = json.load(f)
                    
                    for json_text in json_data['named_entity']:
                        
                        title = json_text['title'][0]['sentence']
                        content = " ".join([i['sentence'] for i in json_text['content']])
                        full_text = title + "\n" + content
                        
                        # json text
                        json_full_text = {'text': full_text + "<|endoftext|>"}
                        
                        save_to_json_file(json_full_text, dst_path=dst_path)
                        
                        # append
                        full_text_list.append(json_full_text)


if __name__ == '__main__':
    main()