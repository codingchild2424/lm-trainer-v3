from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import argparse

def define_args():
    
    p = argparse.ArgumentParser()
    
    p.add_argument("--src_path", type=str)
    p.add_argument("--tgt_path", type=str)
    
    cfg = p.parse_args()
    
    return cfg

def main(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.src_path)
    model = AutoModelForCausalLM.from_pretrained(cfg.src_path, ignore_mismatched_sizes=True)
    
    tokenizer.push_to_hub(cfg.tgt_path)
    model.push_to_hub(cfg.tgt_path)
    

if __name__ == "__main__":
    
    cfg = define_args()
    main(cfg)