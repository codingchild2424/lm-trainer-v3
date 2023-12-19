
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

parser = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=54))
parser.add_argument('--model_name_or_path', type=str, default=None, nargs='?', help="Path to the input model.")
parser.add_argument('--output_dir', type=str, default=None, help='Path to the output folder.')
parser.add_argument("--max-shard-size", type=str, default="1GB", help="Maximum size of a shard in GB or MB.")
parser.add_argument('--bf16', action='store_true', help='Load the model with bfloat16 precision. Requires NVIDIA Ampere GPU.')
args = parser.parse_args()

if __name__ == '__main__':
    #path = args.model_name_or_path
    model_name = args.model_name_or_path

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        low_cpu_mem_usage=True, 
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    out_folder = args.output_dir
    
    model.save_pretrained(
        out_folder, 
        max_shard_size=args.max_shard_size, 
        safe_serialization=True
    )
    tokenizer.save_pretrained(out_folder)