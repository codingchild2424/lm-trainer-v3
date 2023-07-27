#!/usr/bin/env python3
from os.path import basename, isfile
from urllib.error import HTTPError
from urllib.parse import quote
from urllib import request
from json import loads
from sys import argv
import argparse
import json
from tqdm import tqdm

def error(text):
    exit(f'{basename(__file__)}: error: {text}')


def showcodes():
    for code, lang in languages.items():
        print(f'{code} - {lang}')
    exit()


def translator(text, tl, sl):
    try:
        text_json = request.urlopen(f'https://translate.googleapis.com/translate_a/single?client=gtx&sl={sl}&tl={tl}&dt=t&q={quote(text)}').read()
    except HTTPError as Err:
        if Err.code == 400:
            error('bad request, maybe your text is too long')
        else:
            error(repr(Err))

    results = loads(text_json)[0]

    result_list = []

    for phrase in results:
        result_list.append(phrase[0])
        
    total_result = "".join(result_list)
        
    return total_result


def main():
    
    data_path = "../../datasets/raw_datasets/open_orca.json"
    
    result_list = []
    
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # read all lines

    for line in tqdm(lines, desc="Translating"):  # tqdm will display a progress bar
        json_line = json.loads(line)['text']
        # Translation
        trans_result = translator(json_line, tl="ko", sl="auto")
        json_trans_result = json.dumps({"text": trans_result})
        result_list.append(json_trans_result)
    
    dst_path = "../../datasets/pre_datasets/ko_open_orca.json"
    
    with open(dst_path, "w", encoding="utf-8") as f:
        
        pbar = tqdm(total=len(result_list))
        
        for json_str in result_list:
            f.write(json_str + "\n")
            pbar.update()
            
        pbar.close()
    
    return trans_result



if __name__ == '__main__':
    
    trans_result = main()