import json
import pandas as pd
import argparse
import random
from tqdm import tqdm

def instruction_selector(instruction_num):
    
    if instruction_num == 0:
        instuction = " ".join([
            "지시:릴레이 동화 만들기를 합니다.",
            "한 문장씩 번갈아 동화를 만듭니다.",
            "민감한 사회적 문제, 욕설, 위험, 폭력적인 발언을 절대 하지 않습니다.",
            "불필요하게 비슷한 말을 반복하지 않습니다.",
            "높임말이나 반말 중에서 한 가지만을 일관되게 사용합니다.",
            "자, 그럼 이제부터 릴레이 동화 만들기를 시작합니다."
        ])
    elif instruction_num == 1:
        instuction = " ".join([
        "지시: 릴레이 동화창작을 시작합니다.",
        "한 문장씩 차례로 동화를 창작해 나갑니다.",
        "부적절한 언어, 사회적 민감사항, 폭력적이거나 위협적인 내용은 사용하지 않습니다.",
        "의미없는 반복이나 낮은 창의성을 가진 발언을 자제합니다.",
        "말투는 일관성을 유지하며, 높임말이나 반말 중 하나를 선택하여 사용합니다.",
        "이제부터 릴레이 동화창작을 시작합니다."
        ])
    elif instruction_num == 2:
        instuction = " ".join([
        "지시: 이어서 동화를 만드는 시간을 가집니다.",
        "차례대로 한 문장씩 동화를 이어나가는 릴레이 방식을 사용합니다.",
        "폭력적이거나 사회적으로 민감한 주제, 부적절한 언어는 삼가주세요.",
        "반복되는 내용이나 구체성을 떨어뜨리는 문장 사용을 피합니다.",
        "반말 또는 높임말 중 한 가지를 선택하여 일관성 있게 사용하세요.",
        "그럼, 이어서 동화를 만드는 시간을 시작하겠습니다."
        ])
    elif instruction_num == 3:
        instuction = " ".join([
        "지시: 함께 동화를 창작하는 시간을 가져봅시다.",
        "참가자는 차례로 한 문장씩 동화를 이어나갑니다.",
        "사회적으로 민감하거나 문제가 될 수 있는 내용, 욕설이나 위험, 폭력을 유발할 수 있는 발언은 절대 사용하지 않습니다.",
        "불필요한 반복이나 의미 없는 내용도 지양합니다.",
        "말투는 통일시켜 반말 또는 높임말 중 하나만 사용합니다.",
        "함께 동화를 창작하는 시간을 시작합니다."
        ])
    elif instruction_num == 4:
        instuction = " ".join([
        "지시: 릴레이 동화 창작에 도전해봅시다.",
        "각자 차례로 한 문장씩 추가해 동화를 완성합니다.",
        "인격 모독, 사회적 민감 이슈, 폭력적이거나 위협적인 내용은 피합니다.",
        "단조롭게 반복되는 표현이나 낮은 창의성을 가진 발언을 자제하세요.",
        "일관된 말투를 유지하기 위해 높임말이나 반말 중 하나만 사용해주세요.",
        "이제 릴레이 동화 창작에 도전해보겠습니다."
        ])
    elif instruction_num == 5:
        instuction = " ".join([
        "지시: 우리 함께 동화를 써봅시다.",
        "각자 한 문장씩 순서대로 동화를 이어갑니다.",
        "민감한 사회적 이슈, 폭력적이거나 위협적인 언어, 욕설 등의 사용을 피합니다.",
        "동일한 표현의 과도한 반복이나 의미 없는 내용 추가를 자제해주세요.",
        "또한, 말투는 높임말이나 반말 중 한 가지 스타일로 통일합니다.",
        "지금부터 우리 동화를 써봅시다."
        ])
    elif instruction_num == 6:
        instuction = " ".join([
        "지시: 릴레이 방식으로 동화를 만들어봅시다.",
        "참가자 각각이 번갈아 가며 한 문장씩 추가하여 동화를 완성합니다.",
        "사회적으로 민감한 내용, 폭력적이거나 위험한 언어 사용은 절대 금지합니다.",
        "또한, 중복되는 표현이나 낮은 창의성을 가진 문장 사용을 피해주세요.",
        "말투는 높임말이나 반말 중 선택하여 일관성 있게 유지해주세요.",
        "지금부터 릴레이 방식으로 동화를 만들어보겠습니다."
        ])
    elif instruction_num == 7:
        instuction = " ".join([
        "지시: 즐거운 동화 만들기의 시간입니다.",
        "참가자들이 한 문장씩 동화를 만들어 나갑니다.",
        "욕설, 위협, 폭력적인 언어 및 사회적으로 민감한 주제는 절대로 사용하지 않습니다.",
        "반복되는 문장이나 창의성을 떨어뜨리는 발언도 자제해주세요.",
        "높임말이나 반말 중 하나만 선택하여 일관성 있게 사용해주세요.",
        "이제 즐거운 동화 만들기의 시간을 시작합시다."
        ])
    elif instruction_num == 8:
        instuction = " ".join([
        "지시:  동화 작성 릴레이를 실시합니다.",
        "각자 차례로 한 문장씩 동화를 만들어 가는 방식으로 진행됩니다.",
        "폭력적, 위협적, 민감한 사회적 이슈에 대한 내용, 또는 욕설 등은 절대로 사용하지 않습니다.",
        "불필요한 반복이나 고유성이 떨어지는 문장 사용은 피해주세요.",
        "또한, 반말 또는 높임말 중 하나를 선택해 일관성을 유지하세요.",
        "이제부터 동화 작성 릴레이를 시작합니다."
        ])
    elif instruction_num == 9:
        instuction = " ".join([
        "지시: 창작 동화 릴레이를 시작하겠습니다.",
        "각자 차례로 한 문장씩 동화를 이어가는 방식을 사용합니다.",
        "부적절한 언어, 사회적으로 민감한 주제, 폭력적이거나 위협적인 내용은 삼가주세요.",
        "또한, 같은 내용의 반복이나 창의력이 부족한 문장은 사용하지 않도록 해주세요.",
        "반말 또는 높임말 중 하나를 선택하여 일관성 있게 말하도록 해주세요.",
        "창작 동화 릴레이를 시작하겠습니다."
        ])

    return instuction

def define_args():
    
    p = argparse.ArgumentParser()
    
    p.add_argument("--src_path", type=str)
    p.add_argument("--dst_path", type=str)
    
    config = p.parse_args()
    
    return config


def main(cfg):
    
    MASK_TOKEN = "<MASK>"
    
    
    final_instruction_whole_list = []
    
    # open
    with open(cfg.src_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()  # read all lines
        
        for line in tqdm(lines, desc="Processing"):
            
            instruction_text = instruction_selector(random.randint(0, 9))
            
            line_list = json.loads(line)['text'].split('\n')
            
            masking_num = random.randint(0, len(line_list))
            
            final_instruction_list = []
            
            answer = ""
            
            for i, e in enumerate(line_list):
                
                if i == masking_num:
                    sentence = MASK_TOKEN
                    answer = e
                else:
                    sentence = e
            
                sentence = str(i+1) + "막: " + sentence
                
                final_instruction_list.append(sentence)
                
            final_instruction = "\n".join(
                [instruction_text] + \
                final_instruction_list + \
                [MASK_TOKEN + "에 들어갈 가장 적절한 문장을 작성하라.\n문장: " + answer]
                )
                      
            final_instruction_whole_list.append(final_instruction)
            
    dicts = ({"text": text} for text in final_instruction_whole_list)
        
    with open(cfg.dst_path, 'w', encoding='utf-8') as f:
        for d in tqdm(dicts, desc="Processing"):
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
        

if __name__ == "__main__":
    
    cfg = define_args()
    
    main(cfg)