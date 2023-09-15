import chainlit as cl
from chainlit import user_session
from datetime import datetime
import time
from transformers import pipeline, AutoModelForCausalLM
import torch

#############################################################
# session_num
# - session은 user_id를 기준으로, db에서 받아와야 하는 정보임
# - 일단은 default로 0으로 설정
#############################################################
session_num = 0

#############################################################
# avatar_url
#############################################################
chatbot_name = "chatbot"
avatar_url = "https://avatars.githubusercontent.com/u/25720743?s=200&v=4"

#############################################################
# Model
#############################################################

MODEL = "/workspace/Coding/lm-trainer/model_records/koalpaca_sft-v1"
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=MODEL,
    #device=8,
)

def pipe_func(prompt_chain):
    ans = pipe(
        prompt_chain + "\n\n### 답변:",
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    
    msg = ans[0]["generated_text"]
    
    if "###" in msg:
        msg = msg.split("###")[0]
    
    return msg


#############################################################
# history
#############################################################
history=[]

#############################################################
# When a user inputs a message in the UI, the following code is executed:
#############################################################
@cl.on_message
async def main(message: str):
    
    await cl.Avatar(name=chatbot_name, url=avatar_url).send()

    # create a user message
    user_message = {"role": "질문", "content": message}
    # append the user message to the history
    history.append(user_message)
    
    # create a prompt chain
    prompt_chain = "\n".join(
        [f"### {msg['role']}:{msg['content']}" for msg in history]
    )
    # get the bot response
    bot_response = pipe_func(prompt_chain)
    
    # append the bot response to the history
    history.append({"role": "답변", "content": bot_response})

    # send back the final answer
    await cl.Message(content=bot_response, author=chatbot_name).send()