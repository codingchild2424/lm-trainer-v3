import json


def append_to_dst(dst_path, data, language="ko"):
    with open(dst_path, "a") as f:
        
        if language == "ko":
            json.dump(data, f, ensure_ascii=False)
        else:
            json.dump(data, f)
        
        f.write("\n")

def count_processed_data(data_path):
    with open(data_path, "r") as f:
        return sum(1 for line in f)