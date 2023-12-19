
import pandas as pd
import json

src_path = '/workspace/Coding/lm-trainer/datasets/pre_datasets/rm_data/ise_book_data_1000.tsv'
dst_path = '/workspace/Coding/lm-trainer/datasets/pre_datasets/rm_data/ise_book_data_1000.json'

def append_to_dst(dst_path, data, language="ko"):
    with open(dst_path, "a") as f:
        
        if language == "ko":
            json.dump(data, f, ensure_ascii=False)
        else:
            json.dump(data, f)
        
        f.write("\n")
        


def main():
    df = pd.read_csv(src_path, sep='\t')
    
    for i in range(len(df)):
        chosen = df.iloc[i, 0]
        rejected = df.iloc[i, 1]
        
        data = {"chosen": chosen + "<|endoftext|>", "rejected": rejected + "<|endoftext|>"}
        
        append_to_dst(dst_path, data)
        

if __name__ == '__main__':
    main()