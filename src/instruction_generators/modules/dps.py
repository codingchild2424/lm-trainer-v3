
"""
dps is a data preprocessor modules.
"""

import json
import os
from langchain.prompts import PromptTemplate


class DpsModule:
    def __init__(
        self
        ):
        pass
    
    def preprocess_with_prompt_chain_generator(
        self, 
        data,
        prompt_chain_maker
        ):
        
        prompt_chain_result_list = []
        
        for i in data:
            # 지금은 바로 prompt에 대화만 넣을 것이니 이렇게 작업하기
            dialogues = "\n".join(
                list(i['talk']['content'].values())
            )
            
            prompt_chain_result = prompt_chain_maker(dialogues)
            
            prompt_chain_result_list.append(prompt_chain_result)
            
        return prompt_chain_result_list
    
    def preprocess(
        self, 
        data,
        prompt_chain_maker
        ):
        
        prompt_chain_result_list = []
        
        for i in data:
            # 지금은 바로 prompt에 대화만 넣을 것이니 이렇게 작업하기
            dialogues = "\n".join(
                list(i['talk']['content'].values())
            )
            
            prompt_chain_result = prompt_chain_maker(dialogues)
            
            prompt_chain_result_list.append(prompt_chain_result)
            
        return prompt_chain_result_list


    def postprocess(self, data):
        
        data_list = []
        
        for i in data.split("\n"):
            if i == "":
                continue
            else:
                try:
                    data_list.append(
                        { i.split(":")[0] : i.split(":")[1] }
                    ) 
                except:
                    print("[70] postprocess error", i)
                    continue
                
        data_list_dict = {
            "text": data_list
        }
        
        return data_list_dict