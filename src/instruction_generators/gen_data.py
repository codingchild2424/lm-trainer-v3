#####################################################
# Library
#####################################################
import os
import json
from tqdm import tqdm
import multiprocessing
import time

#####################################################
# Modules
#####################################################
from modules.gpt_call import gpt_call
from modules.define_argparser import define_argparser

from modules.dps import DpsModule
from modules.prompt_chain_maker import PromptChainMaker


DPS = DpsModule()

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
    
def process_data_chunk(chunk):
    results = []
    for i in chunk:
        gpt_result = gpt_call(
            model_name="gpt-4",
            prompt=str(i)
        )
        post_gpt_result = DPS.postprocess(gpt_result)
        results.append(post_gpt_result)
    return results

#####################################################
# Main
#####################################################
def main(cfg):
    # Check if the dst_path exists
    start_index = 0
    if os.path.exists(cfg.dst_path):
        # Count processed data and set the start index to continue from there
        start_index = count_processed_data(cfg.dst_path)

    # file type check
    if cfg.src_path != None:
        if cfg.src_path.split(".")[-1] != "json":
            raise ValueError("src_path should be json file")
        # json file load
        elif cfg.src_path.split(".")[-1] == "json":
            with open(cfg.src_path, "r") as f:
                json_data = json.load(f)
        else:
            json_data = None
    
    #####################################################
    # Preprocess
    #####################################################
    if cfg.preprocess_type == "normal":
        PCM = PromptChainMaker(
            input_variables=["dialogues"],
            prompt_template_path=cfg.prompt_template_path
            )
        prompt_chain_result_list = DPS.preprocess(
            data = json_data,
            )
    elif cfg.preprocess_type == "prompt_chain":
        PCM = PromptChainMaker(
            input_variables=["dialogues"],
            prompt_template_path=cfg.prompt_template_path
            )
        prompt_chain_result_list = DPS.preprocess_with_prompt_chain_generator(
            data = json_data,
            prompt_chain_maker = PCM.prompt_chain_maker
            )
    else:
        raise ValueError("preprocess_type should be normal or prompt_chain")
    
    # # Skip already processed data
    data = prompt_chain_result_list[start_index:]
    
    # # Determine the chunk size based on RPM
    
    if cfg.multi_processing == True:
        # Given RPM is 200
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            for i in tqdm(range(0, len(data), cfg.multi_processing_chunk_size)):
                chunk = data[i:i+chunk_size]
                results = pool.map(process_data_chunk, [chunk])
                for res in results:
                    for item in res:
                        append_to_dst(cfg.dst_path, item)
    elif cfg.multi_processing == False:
        for i in tqdm(data):
        
            gpt_result = gpt_call(
                model_name="gpt-4",
                prompt=str(i)
                )
            
            post_gpt_result = DPS.postprocess(gpt_result)
            
            append_to_dst(
                dst_path=cfg.dst_path,
                data=post_gpt_result
            )
    else:
        raise ValueError("multi_processing should be True or False")
        
if __name__ == '__main__':
    cfg = define_argparser()
    main(cfg)