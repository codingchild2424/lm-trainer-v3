from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig, 
    HfArgumentParser, 
    TrainingArguments, 
    AutoTokenizer,
    pipeline
)
from trl import (
    PPOConfig,
    AutoModelForCausalLMWithValueHead,
    PPOTrainer
)

from tqdm import tqdm

@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine with SFTTrainer
    """
    model_name_or_path: Optional[str] = field(
        default="facebook/opt-350m", metadata={"help": "the model name"}
    )
    train_file: Optional[str] = field(
        default="timdettmers/openassistant-guanaco", metadata={"help": "the dataset name"}
    )
    dataset_text_field: Optional[str] = field(
        default="text", metadata={"help": "the text field of the dataset"}
    )
    log_with: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=64, metadata={"help": "the batch size"}
    )
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    torch_dtype: Optional[str] = field(
        default="float16", metadata={"help": "the dtype of the model"}
    )
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_peft: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use PEFT or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    peft_lora_r: Optional[int] = field(
        default=64, metadata={"help": "the r parameter of the LoRA adapters"}
    )
    peft_lora_alpha: Optional[int] = field(
        default=16, metadata={"help": "the alpha parameter of the LoRA adapters"}
    )
    logging_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of logging steps"}
    )
    use_auth_token: Optional[bool] = field(
        default=True, metadata={"help": "Use HF auth token to access the model"}
    )
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "the number of training epochs"}
    )
    max_steps: Optional[int] = field(
        default=-1, metadata={"help": "the number of training steps"}
    )
    save_steps: Optional[int] = field(
        default=100, metadata={"help": "Number of updates steps before two checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )
    push_to_hub: Optional[bool] = field(
        default=False, metadata={"help": "Push the model to HF Hub"}
    )
    hub_model_id: Optional[str] = field(
        default=None, metadata={"help": "The name of the model on HF Hub"}
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "DeepSpeed training configuration file"}
    )
    validatoin_split_percentage: Optional[int] = field(
        default=5, metadata={"help": "The percentage of the train set used as validation set"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "The cache directory"}
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=True, metadata={"help": "Use fast tokenizer"}
    )
    model_revision: Optional[str] = field(
        default="main", metadata={"help": "The revision of the model"}
    )
    use_auth_token: Optional[bool] = field(
        default=False, metadata={"help": "Use HF auth token to access the model"}
    )
    bf16: Optional[bool] = field(
        default=False, metadata={"help": "Use bfloat16 precision"}
    )
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "Use fp16 precision"}
    )
    reward_model_name_or_path: Optional[str] = field(
        default="facebook/bart-large", metadata={"help": "the model name"}
    )




def main():
    parser = HfArgumentParser(ScriptArguments)
    
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # Step 1: Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError("You can't load the model in 8 bits and 4 bits at the same time")
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": Accelerator().local_process_index}
        torch_dtype = torch.bfloat16
    else:
        device_map = None
        quantization_config = None
        
        # torch_dtype
        if script_args.torch_dtype == "float16":
            torch_dtype = torch.float16
        elif script_args.torch_dtype == "float32":
            torch_dtype = torch.float32
        elif script_args.torch_dtype == "bfloat16":
            # for Amphere GPU (A100, A6000...)
            # if you have trouble which lr doesn't decrease, try to use bfloat16
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = None
    
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch_dtype,
        use_auth_token=script_args.use_auth_token,
    )
    
    # Step 2: Load the tokenizer
    tokenizer_kwargs = {
        "cache_dir": script_args.cache_dir,
        "use_fast": script_args.use_fast_tokenizer,
        "revision": script_args.model_revision,
        "use_auth_token": True if script_args.use_auth_token else None,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path,
        **tokenizer_kwargs,
        )
    
    tokenizer.pad_token = tokenizer.eos_token
    
    # Step 3: Load the dataset
    # Currently, you can only use jsonl file.
    # If you want to use other file, you should modify this code.
    if script_args.train_file.endswith(".json"):
        dataset = load_dataset(
            "json",
            data_files=script_args.train_file,
            split='train',
            )
    else:
        raise ValueError("You should use jsonl.")
    
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["text"])
        sample['query'] = tokenizer.decode(sample["input_ids"])
        
        return sample

    dataset = dataset.map(tokenize, batched=False)
    
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
    
    # Step 4. PPO Config and PPO Trainer
    ppo_config = PPOConfig(
        model_name=script_args.model_name_or_path,
        learning_rate=script_args.learning_rate,
    )
    
    ppo_trainer = PPOTrainer(
        model=model,
        # # the second model serves as a reference to calculate the KL-divergence from the starting point.
        ref_model=ref_model,
        config=ppo_config,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )
    
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    # 5. Load reward model
    
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"
    
    reward_model = pipeline(
        "text-classification",
        model=script_args.reward_model_name_or_path,
        tokenizer=script_args.reward_model_name_or_path,
        device=device,
    )
    
    
    # 6. Train 
    for epoch, batch in tqdm(enumerate(ppo_trainer.get_train_dataloader())):
        
        query_tensors = batch["input_ids"]

        #### Get response from model
        response_tensors = []
        for query in query_tensors:
            gen_len = output_length_sampler()
            generation_kwargs["max_new_tokens"] = gen_len
            response = ppo_trainer.generate(query, **generation_kwargs)
            response_tensors.append(response.squeeze()[-gen_len:])
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        #### Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards)
        #### Save model
        ppo_trainer.save_model(script_args.output_dir)


if __name__ == '__main__':
    main()