from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request, error
from urllib.parse import quote
from json import loads, dumps
from tqdm import tqdm
import json

MAX_WORKERS = 10  # Set the number of threads you want to use.

def translator(text, tl, sl):
    try:
        text_json = request.urlopen(f'https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl={tl}&dt=t&q={quote(text)}').read()
    except error.HTTPError as Err:
        if Err.code == 400:
            print('bad request, maybe your text is too long')
        else:
            print(repr(Err))

    results = loads(text_json)[0]

    result_list = []

    for phrase in results:
        result_list.append(phrase[0])
        
    total_result = "".join(result_list)
        
    return total_result

def translate_line(line):
    json_line = json.loads(line)['text']
    # Translation
    trans_result = translator(json_line, tl="ko", sl="auto")
    json_trans_result = json.dumps({"text": trans_result})
    
    return json_trans_result

def main():
    
    data_path = "../../datasets/raw_datasets/open_orca.json"
    
    result_list = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # read all lines

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Start the load operations and mark each future with its line
        future_to_line = {executor.submit(translate_line, line): line for line in lines}
        for future in tqdm(as_completed(future_to_line), total=len(lines), desc="Translating"):
            line = future_to_line[future]
            try:
                data = future.result()
            except Exception as exc:
                print('%r generated an exception: %s' % (line, exc))
            else:
                result_list.append(data)
    
    dst_path = "../../datasets/pre_datasets/ko_open_orca.json"
    
    with open(dst_path, "w", encoding="utf-8") as f:
        for json_str in tqdm(result_list, desc="Writing"):
            f.write(json_str + "\n")
            
            
if __name__ == '__main__':
    main()
