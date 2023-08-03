from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def define_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_name', type=str)
    
    cfg = p.parse_args()
    
    return cfg

def generate_text(prompt, model, tokenizer, max_length=50, temperature=1.0):
    # Encode the prompt text to tensor
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text from the model
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature
    )
    
    # Decode the generated text back to a string
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text

def main(cfg):
    
    model_name_or_path = cfg.model_name
    
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    
    prompt = input("Enter a prompt: ")
    generated_text = generate_text(prompt, model, tokenizer)

    print("Generated text:", generated_text)
    
if __name__ == "__main__":
    cfg = define_argparser()
    main(cfg)