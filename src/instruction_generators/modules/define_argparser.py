import argparse


#####################################################
# Argparser
#####################################################
def define_argparser():
    p = argparse.ArgumentParser()
    
    # prompt_template_path
    p.add_argument('--prompt_template_path', type=str, required=True)
    # seed path
    p.add_argument('--seed_path', type=str, default=None)
    
    p.add_argument('--input_variables', type=list, default=[None])

    # model_file_name
    p.add_argument('--src_path', type=str, default=None)
    p.add_argument('--dst_path', type=str, required=True)
    p.add_argument('--model_name', type=str, default="gpt-4")
    p.add_argument('--dps_name', type=str, default="dps")
    
    p.add_argument('--preprocess_type', type=str, default="prompt_chain")
    p.add_argument('--multi_processing', type=bool, default=False)
    p.add_argument('--multi_processing_chunk_size', type=int, default=200)
    
    
    
    
    
    
    
    
    cfg = p.parse_args()

    return cfg