from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import pipeline

model_name_or_path = "/workspace/Coding/lm-trainer/model_records/polyglot-ko-5.8b-title-epoch_6"

#using pipeline for inference with GPU
generator = pipeline('text-generation', model=model_name_or_path, device=0)

# Tokenize the input text
input_text = "\n\n".join([
    "### 이야기:내 나무 숲에 들어서자, 작은 새 친구가 반겨왔다. 나는 작은 새에게 반갑게 인사했다.",
    "### 제목:"
])

output = generator(input_text, max_length=100, num_return_sequences=1)[0]['generated_text']

print(output)


input_text = "\n\n".join([
    "### 이야기:사라와 벤은 엄마를 위해 그릇을 장식하고 싶었습니다. 그들은 부엌에서 큰 그릇과 페인트와 붓을 발견했습니다.",
    "### 제목:"
])

output = generator(input_text, max_length=100, num_return_sequences=1)[0]['generated_text']

print(output)


input_text = "\n\n".join([
    "### 이야기:그들은 그릇과 페인트를 뒤뜰로 가져가 탁자 위에 올려 놓았습니다.\n\"색깔로 그릇을 예쁘게 꾸며보자\" 사라가 말했다.\n\"좋아, 꽃을 그릴게.\" 벤이 말했다.\n그들은 다른 색으로 그릇을 칠하기 시작했습니다.",
    "### 제목:"
])

output = generator(input_text, max_length=100, num_return_sequences=1)[0]['generated_text']

print(output)