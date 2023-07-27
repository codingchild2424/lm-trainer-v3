from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib import request, error
from urllib.parse import quote
from json import loads
import json
import threading
import os
from tqdm import tqdm  # progress bar

MAX_WORKERS = 10  # Set the number of threads you want to use.
lock = threading.Lock()

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

def translate_line(line, output_file, progress_bar):
    original_text = json.loads(line)['text']

    # Translation
    trans_result = translator(original_text, tl="ko", sl="auto")
    json_trans_result = json.dumps({"text": trans_result}, ensure_ascii=False) # Add the argument ensure_ascii=False

    # Lock the thread and write to the file
    with lock:
        output_file.write(json_trans_result + "\n")
        progress_bar.update()

def main():
    
    data_path = "../../datasets/raw_datasets/open_orca.json"
    dst_path = "../../datasets/pre_datasets/ko_open_orca.json"
    
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # read all lines

    num_completed_lines = 0

    if os.path.exists(dst_path):
        with open(dst_path, 'r', encoding='utf-8') as f:
            for line in f:
                num_completed_lines += 1

    with open(dst_path, "a", encoding="utf-8") as output_file:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            progress_bar = tqdm(total=len(lines) - num_completed_lines, desc="Translating")
            # Start the load operations and mark each future with its line
            future_to_line = {executor.submit(translate_line, line, output_file, progress_bar): line for line in lines[num_completed_lines:]}
            for future in as_completed(future_to_line):
                line = future_to_line[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print('%r generated an exception: %s' % (line, exc))
            progress_bar.close()

if __name__ == '__main__':
    main()
