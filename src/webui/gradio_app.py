import gradio as gr
from transformers import AutoTokenizer, AutoModelWithLMHead

# Load the GPT-NeoX model and its tokenizer.
# You can replace 'EleutherAI/gpt-neox-1.3B' with the appropriate model name if different.
model_name = '/workspace/Coding/lm-trainer/src/model_results/polyglot-5.8b-total-385159-epoch1'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelWithLMHead.from_pretrained(model_name)

def generate_text(prompt, _):  # The second argument captures the button's value but doesn't use it
    """Generate text using GPT-2."""
    inputs = tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Create Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=["textarea", "button"],
    outputs="textarea",
    title="GPT-NeoX Text Generator",
    description="Generate text using GPT-NeoX by EleutherAI."
)

iface.launch()
