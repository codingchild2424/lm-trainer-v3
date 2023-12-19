import pandas as pd
import os
import json
from tqdm import tqdm


def save_to_json_file(json_response, dst_path: str):
    with open(dst_path, 'a') as outfile:
        json.dump(json_response, outfile, ensure_ascii=False)
        outfile.write('\n')


def main():
    data_folder_path = "/workspace/datasets/NLP/raw_datasets/ai_hub/029.대규모 구매도서 기반 한국어 말뭉치 데이터/01.데이터/4.Sample/sample/라벨링데이터/000"
    dst_path = "/workspace/Coding/lm-trainer/datasets/pre_datasets/cl_data/aihub_book.json"

    file_path_list = [os.path.join(data_folder_path, file_name) for file_name in os.listdir(data_folder_path)]

    with tqdm(file_path_list) as pbar:
        pbar.set_description("Processing")

        for file_path in pbar:
            with open(file_path, 'r', encoding='utf-8') as f:
                # json file
                json_data = json.load(f)
                
                # is json_data has key name 'paragraphs'
                if 'paragraphs' in json_data.keys():
                    for idx, paragraph in enumerate(json_data['paragraphs']):
                
                        paragraph_text = " ".join([i['text'] for i in paragraph['sentences']])
                        
                        # json text
                        json_full_text = {'text': paragraph_text + "<|endoftext|>"}
                        
                        save_to_json_file(json_full_text, dst_path=dst_path)
                else:
                    continue


if __name__ == '__main__':
    main()