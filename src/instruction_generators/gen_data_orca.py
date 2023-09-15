#####################################################
# Library
#####################################################
import os
import json
from tqdm import tqdm
import multiprocessing
import time
import jsonlines

#####################################################
# Modules
#####################################################
from modules.gpt_call import gpt_call
from modules.define_argparser import define_argparser

from modules.dps import DpsModule
from modules.prompt_chain_maker import PromptChainMaker
from modules.utils import append_to_dst, count_processed_data

DPS = DpsModule()

#####################################################
# Main
#####################################################
def main(cfg):
    # Check if the dst_path exists
    start_index = 0
    if os.path.exists(cfg.dst_path):
        # Count processed data and set the start index to continue from there
        start_index = count_processed_data(cfg.dst_path)
        
    #####################################################
    # Seed file load (jsonl)
    #####################################################
    if cfg.seed_path != None:
        # json file load
        if cfg.seed_path.split(".")[-1] == "parquet":
            seed_data = pd.read_parquet(cfg.seed_path)
    else:
        raise ValueError("seed_path should not be None")
            
    #####################################################
    # Prompt Template file load (txt)
    # Variables in Prompt Template should be like this: {var0}, {var1}, {var2}
    #####################################################
    if cfg.prompt_template_path != None:
        if cfg.prompt_template_path.split(".")[-1] == "txt":
            with open(cfg.prompt_template_path, "r") as f:
                prompt_template = f.read()
        else:
            raise ValueError("prompt_template_path should be txt file")
    else:
        raise ValueError("prompt_template_path should not be None")
    
    #####################################################
    # PromptChainMaker
    #####################################################
    
    # Input_variables should be less than 4
    # If you want to use more than 3 variables, you should modify PromptChainMaker
    
    PCM = PromptChainMaker(
        input_variables=cfg.input_variables,
        seed_data=seed_data,
        prompt_template=prompt_template
    )
    
    for i in tqdm(data):
        
        prompt_template_format = PCM.prompt_chain_maker(
            var0=i['talk']['content']['user'],
        )
        
        gpt_result = gpt_call(
            model_name=cfg.model_name,
            prompt=prompt_template_format
            )
        
        post_gpt_result = DPS.postprocess(gpt_result)
        
        append_to_dst(
            dst_path=cfg.dst_path,
            data=post_gpt_result
        )
    
if __name__ == '__main__':
    cfg = define_argparser()
    main(cfg)