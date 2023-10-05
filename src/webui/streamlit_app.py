import streamlit as st
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load the GPT-NeoX model and its tokenizer.
# You can replace 'EleutherAI/gpt-neox-1.3B' with the appropriate model name if different.
model_name = '/workspace/Coding/lm-trainer/src/model_results/polyglot-12.8b-total-364004-v1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

def generate_text(prompt):
    """Generate text using GPT-NeoX."""
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

st.title("GPT-NeoX Text Generator")
st.write("Enter a prompt below and get the completion from GPT-NeoX:")

user_input = st.text_area("Prompt", "Once upon a time...")
if st.button("Generate"):
    with st.spinner("Generating..."):
        generated_output = generate_text(user_input)
    st.text_area("Generated Text", generated_output)

if __name__ == "__main__":
    st.run()
