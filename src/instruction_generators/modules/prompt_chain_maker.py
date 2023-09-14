from langchain.prompts import PromptTemplate


class PromptChainMaker:
    def __init__(
        self,
        input_variables: list, 
        prompt_template_path: str
        ):
        
        with open(prompt_template_path, "r") as f:
            self.prompt_template = f.read()
            
        self.input_variables = input_variables
    
    def prompt_chain_maker(
        self,
        dialogues: str
        ):
        
        prompt_template_result = PromptTemplate(
            input_variables=self.input_variables,
            template=self.prompt_template
        )
        prompt_template_format = prompt_template_result.format(
            dialogues=dialogues
        )
        
        return prompt_template_format