from langchain.prompts import PromptTemplate


class PromptChainMaker:
    def __init__(
        self,
        input_variables: list, 
        seed_data: list,
        prompt_template: str,
        ):
        self.seed_data = seed_data
        self.prompt_template = prompt_template
        self.input_variables = input_variables
    
    def prompt_chain_maker_orca(
        self,
        var0=None,
        var1=None,
        var2=None,
        ):
        
        if len(self.input_variables) == 0:
            prompt_template_foramt = self.prompt_template
            
        elif len(self.input_variables) == 1:
            prompt_template_result = PromptTemplate(
                input_variables=self.input_variables,
                template=self.prompt_template
            )
            prompt_template_format = prompt_template_result.format(
                var0=var0
            )
            
        elif len(self.input_variables) == 2:
            prompt_template_result = PromptTemplate(
                input_variables=self.input_variables,
                template=self.prompt_template
            )
            prompt_template_format = prompt_template_result.format(
                var0=var0,
                var1=var1,
            )
            
        elif len(self.input_variables) == 3:
            prompt_template_result = PromptTemplate(
                input_variables=self.input_variables,
                template=self.prompt_template
            )
            prompt_template_format = prompt_template_result.format(
                var0=var0,
                var1=var1,
                var2=var2
            )
        # If input_variables is more than 3, raise ValueError
        else:
            raise ValueError("input_variables should be less than 3")
        
        return prompt_template_format