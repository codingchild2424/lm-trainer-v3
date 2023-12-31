# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from dataclasses import dataclass, field
from typing import Optional

import torch

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser
from transformers import TrainingArguments
from trl import RewardTrainer #, RewardConfig


tqdm.pandas()

@dataclass
class RewardConfig(TrainingArguments):
    """
    RewardConfig collects all training arguments related to the [`RewardTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, *optional*, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        gradient_checkpointing (`bool`, *optional*, defaults to `True`):
                If True, use gradient checkpointing to save memory at the expense of slower backward pass.
    """

    max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator."
        },
    )
    # gradient_checkpointing: Optional[bool] = field(
    #     default=False,
    #     metadata={
    #         "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
    #     },
    # )



@dataclass
class ScriptArguments:
    """
    Hyperparameters to fine-tune a reward model on a given dataset with the `RewardTrainer`.
    """

    model_name_or_path: Optional[str] = field(default="facebook/opt-350m", metadata={"help": "the model name"})
    train_file: Optional[str] = field(default="Anthropic/hh-rlhf", metadata={"help": "the dataset name"})
    dataset_text_field: Optional[str] = field(default="text", metadata={"help": "the text field of the dataset"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    logging_steps: Optional[int] = field(default=500, metadata={"help": "the number of update steps between two logs"})
    eval_split: Optional[str] = field(
        default="none", metadata={"help": "the dataset split to evaluate on; default to 'none' (no evaluation)"}
    )
    learning_rate: Optional[float] = field(default=1.41e-5, metadata={"help": "the learning rate"})
    per_device_train_batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    num_train_epochs: Optional[int] = field(default=1, metadata={"help": "the number of training epochs"})
    seq_length: Optional[int] = field(default=512, metadata={"help": "Input sequence length"})
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    #gradient_checkpointing: Optional[bool] = field(default=True, metadata={"help": "Enable gradient checkpointing"})
    load_in_8bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 8 bits precision"})
    load_in_4bit: Optional[bool] = field(default=False, metadata={"help": "load the model in 4 bits precision"})
    use_peft: Optional[bool] = field(default=False, metadata={"help": "Wether to use PEFT or not to train adapters"})
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable `trust_remote_code`"})
    output_dir: Optional[str] = field(default="output", metadata={"help": "the output directory"})
    
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "DeepSpeed training configuration file"}
    )
    bf16: Optional[bool] = field(
        default=False, metadata={"help": "Use bfloat16 precision"}
    )
    fp16: Optional[bool] = field(
        default=False, metadata={"help": "Use fp16 precision"}
    )
    torch_dtype: Optional[str] = field(
        default="float16", metadata={"help": "torch dtype"}
    )
    save_total_limit: Optional[int] = field(
        default=10, metadata={"help": "Limits total number of checkpoints."}
    )
    
    


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

model = AutoModelForSequenceClassification.from_pretrained(
    script_args.model_name_or_path,
    quantization_config=quantization_config,
    device_map=device_map,
    trust_remote_code=script_args.trust_remote_code,
    torch_dtype=torch_dtype,
    num_labels=1,
)

# Step 2: Load the dataset and pre-process it
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)

if script_args.train_file.endswith(".csv"):
    train_dataset = load_dataset("csv", data_files=script_args.train_file, split="train")
elif script_args.train_file.endswith(".json"):
    train_dataset = load_dataset("json", data_files=script_args.train_file, split="train")
else:
    train_dataset = load_dataset(script_args.train_file, split="trai")


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(examples):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
        tokenized_chosen = tokenizer(chosen, truncation=True)
        tokenized_rejected = tokenizer(rejected, truncation=True)

        new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
        new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
        new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
        new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

    return new_examples


# Preprocess the dataset and filter out examples that are longer than script_args.max_length
train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    num_proc=4,
)
train_dataset = train_dataset.filter(
    lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
    and len(x["input_ids_rejected"]) <= script_args.seq_length
)

if script_args.eval_split == "none":
    eval_dataset = None
else:
    eval_dataset = load_dataset(script_args.train_file, split=script_args.eval_split)

    eval_dataset = eval_dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
        and len(x["input_ids_rejected"]) <= script_args.seq_length
    )


# Step 3: Define the training arguments
training_args = RewardConfig(
    output_dir=script_args.output_dir,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    #gradient_checkpointing=script_args.gradient_checkpointing,
    learning_rate=script_args.learning_rate,
    report_to="wandb" if script_args.log_with == "wandb" else None,
    remove_unused_columns=False,
    optim="adamw_torch",
    logging_steps=script_args.logging_steps,
    evaluation_strategy="steps" if script_args.eval_split != "none" else "no",
    max_length=script_args.seq_length,
    deepspeed=script_args.deepspeed, # deepspeed option
    bf16=script_args.bf16, # bf16 mixed-precision
    fp16=script_args.fp16,
    save_total_limit=script_args.save_total_limit,
)

# Step 4: Define the LoraConfig
if script_args.use_peft:
    peft_config = LoraConfig(r=16, lora_alpha=16, bias="none", task_type="SEQ_CLS", modules_to_save=["scores"])
else:
    peft_config = None

# Step 5: Define the Trainer
trainer = RewardTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model(script_args.output_dir)