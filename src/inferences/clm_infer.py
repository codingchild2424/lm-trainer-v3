from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def define_argparser():
    
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_name', type=str)
    p.add_argument('--prompt', type=str)
    
    cfg = p.parse_args()
    
    return cfg

def generate_text(prompt, model, tokenizer, max_length=200, temperature=1.0):
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        ignore_mismatched_sizes=True
        ).to(device)
    
    prompt = " ".join([
        "지시:릴레이 동화 만들기를 합니다.",
        "한 문장씩 번갈아 동화를 만듭니다.",
        "민감한 사회적 문제, 욕설, 위험, 폭력적인 발언을 절대 하지 않습니다.",
        "불필요하게 비슷한 말을 반복하지 않습니다.",
        "높임말이나 반말 중에서 한 가지만을 일관되게 사용합니다.",
        "자, 그럼 이제부터 릴레이 동화 만들기를 시작합니다.",
        "1막: 내 나무 숲으로 동생과 함께 여행을 떠났다.",
        "2막:아름다운 식물과 멋진 동물들이 많이 있었다.",
        "3막:"
    ])
    generated_text = generate_text(prompt, model, tokenizer)

    print("Generated text:", generated_text)
    
if __name__ == "__main__":
    cfg = define_argparser()
    main(cfg)