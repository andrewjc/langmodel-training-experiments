# BEGIN COLAB CONFIG POINT

jsonFilename = "alpaca_data_new.json"
# create a new H5 file
train_h5_filename = 'alpaca_data_train.h5'
test_h5_filename = 'alpaca_data_test.h5'
useT5 = True
forceRebuildDataset = True  # dataset cache not working as the tokenizer can't be detached

# END COLAB CONFIG POINT

import json
import os
from dataclasses import dataclass
from typing import List, Dict

import datasets

import h5py
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn as nn
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from tqdm import tqdm

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollator

CUTOFF_LEN = 256  # Longer sequences require more memory and are slower to train
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0.2  # % of the dataset to use for validation

# If using ROCM on AMD GPU still detects as a cuda device
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params}  all params: {all_param}  trainable%: {100 * trainable_params / all_param}"
    )


class CastOutputToFloat(nn.Sequential):
    def forward(self, x): return super().forward(x).to(torch.float32)


if useT5:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

else:
    tokenizer = LlamaTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", add_eos_token=True,
    )
    tokenizer.pad_token_id = 0

    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        # "/content/drive2/MyDrive/model_tune/llama-7b-pre",
        load_in_8bit=True,
        device_map="auto",
    )

    # model.save_pretrained("/content/drive2/MyDrive/model_tune/llama-7b-pre")

    model = prepare_model_for_int8_training(model)

config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    # target_modules=["q_proj", "v_proj"],
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

for param in model.parameters():
    param.requires_grad = False  # freeze the model - train adapters later
    if param.ndim == 1:
        # cast the small parameters (e.g. layernorm) to fp32 for stability
        param.data = param.data.to(torch.float32)

model.gradient_checkpointing_enable()  # reduce number of stored activations
model.enable_input_require_grads()

model.lm_head = CastOutputToFloat(model.lm_head)

model = get_peft_model(model, config)
print_trainable_parameters(model)


def generate_prompt_string(entry):
    # sorry about the formatting disaster gotta move fast
    if entry["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
    ### Instruction:
    {entry["instruction"]}
    ### Input:
    {entry["input"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
    ### Instruction:
    {entry["instruction"]}"""


def generate_X(data_point):
    return generate_prompt_string(data_point)


# Use </s> to indicate end of sentence, inference can check for this string and stop generating
def generate_Y(data_point):
    return f"""{data_point["output"]} </s>"""


def tokenize(text, text_validate):
    return tokenizer(
        text=text,
        text_target=text_validate,
        truncation=True,
        max_length=CUTOFF_LEN + 1,
        padding="max_length",
    )


# load the JSON data from the file
data = datasets.Dataset.from_json(jsonFilename, keep_in_memory=True)

train_val = data.train_test_split(
    test_size=VAL_SET_SIZE, shuffle=True, seed=42
)
train_dataset = train_val["train"]
test_dataset = train_val["test"]

if forceRebuildDataset:
    train_dataset = train_dataset.shuffle().map(lambda x: (generate_X(x), generate_Y(x)),
                                                keep_in_memory=True,
                                                load_from_cache_file=False).map(
        tokenize,
        batched=True,
        keep_in_memory=True,
        load_from_cache_file=False
    )
    test_dataset = test_dataset.shuffle().map(lambda x: (generate_X(x), generate_Y(x)),
                                              keep_in_memory=True,
                                              load_from_cache_file=False).map(
        tokenize,
        batched=True,
        keep_in_memory=True,
        load_from_cache_file=False
    )

    columns = ['input_ids', 'attention_mask', 'labels']
    train_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)

    # write out the data to disk so we can load it faster later
    train_dataset.save_to_disk("train_data.checkpoint")
    test_dataset.save_to_disk("val_data.checkpoint")
else:
    train_dataset.load_from_disk("train_data.checkpoint")
    test_dataset.load_from_disk("val_data.checkpoint")

BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = 3
EPOCHS = 5
LEARNING_RATE = 1e-4

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        dataloader_num_workers=3,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=True,
        logging_steps=20,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        eval_steps=200,
        save_steps=200,
        output_dir=outputDir,
        save_total_limit=3,
        load_best_model_at_end=True,
        # label_smoothing_factor=0.1,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False

old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
).__get__(model, type(model))

trainer.train()

model.save_pretrained("/content/drive2/MyDrive/lora-alpaca.pt")

print("\n If there's a warning about missing keys above, please disregard :)")
