import argparse


#####################################################
# Argparser
#####################################################
def define_argparser():
    p = argparse.ArgumentParser()

    # model_file_name
    p.add_argument('--src_path', type=str, required=True)
    p.add_argument('--dst_path', type=str, required=True)
    p.add_argument('--model_name', type=str, default="gpt-4")
    p.add_argument('--dps_name', type=str, default="dps")
    p.add_argument('--prompt_template_path', type=str, default="/workspace/Coding/instruction_data_generator/src/prompt_templates/counseling.txt")
    
    cfg = p.parse_args()

    return cfg