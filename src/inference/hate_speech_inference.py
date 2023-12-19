import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Step 1: Loading the Model and Tokenizer
model_path = '/workspace/Coding/lm-trainer/model_records/hate_speech_koelectra_v2'  # Adjust this to wherever you've saved your trained model

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Ensure the model is in evaluation mode
model.eval()

# If you have a GPU, put the model on the GPU
if torch.cuda.is_available():
    model = model.cuda()

# Step 2: Function for Classification
def classify_text(text):
    # Tokenize input and get output from the model
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    
    # If you have a GPU, move the input tensors to the GPU
    if torch.cuda.is_available():
        for key in inputs:
            inputs[key] = inputs[key].cuda()

    with torch.no_grad():
        logits = model(**inputs).logits

    # Get predicted label from logits
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    return "CLEAN" if predicted_class_idx == 0 else "UNCLEAN"

# Step 3: Test the Function
sample_text = "ㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅㅂㅅ"
print(f"'{sample_text}' is classified as: {classify_text(sample_text)}")

sample_text = "오늘은 참 아름답군요."
print(f"'{sample_text}' is classified as: {classify_text(sample_text)}")

sample_text = "두 명의 여성이 서로 키스를 했다."
print(f"'{sample_text}' is classified as: {classify_text(sample_text)}")

sample_text = "참치는 참 맛있어."
print(f"'{sample_text}' is classified as: {classify_text(sample_text)}")