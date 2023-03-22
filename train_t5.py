
jsonFilename = "tiny_json.json"
pretrainedModelPath = "t5-large"

# create a new H5 file
train_h5_filename = './alpaca_data_train.h5'
test_h5_filename = 'alpaca_data_test.h5'
useT5 = True #Use t5 or use llama

import json
import os
from dataclasses import dataclass
from typing import List, Dict

import datasets

import h5py
import numpy as np
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from peft import (
    prepare_model_for_int8_training,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
)
from tqdm import tqdm

import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, DataCollator

BATCH_SIZE = 64
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
EPOCHS = 5  # we don't need 3 tbh
LEARNING_RATE = 1e-4  # the Karpathy constant
CUTOFF_LEN = 256  # 256 accounts for about 96% of the data
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
VAL_SET_SIZE = 0.2

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

if useT5:
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

else:
    tokenizer = LlamaTokenizer.from_pretrained(
        "decapoda-research/llama-7b-hf", add_eos_token=True,
    )
    tokenizer.pad_token_id = 0  # unk. we want this to be different from the eos token

    model = LlamaForCausalLM.from_pretrained(
        "decapoda-research/llama-7b-hf",
        load_in_8bit=True,
        device_map="auto",
    )

    model.save_pretrained(pretrainedModelPath)

    model = prepare_model_for_int8_training(model)

    config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)


def generate_X(data_point):
    # sorry about the formatting disaster gotta move fast
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
{data_point["instruction"]}"""


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

if True:
    train_dataset = train_dataset.shuffle().map(lambda x: tokenize(generate_X(x), generate_Y(x)),
                                                keep_in_memory=True,
                                                load_from_cache_file=False)
    test_dataset = test_dataset.shuffle().map(lambda x: tokenize(generate_X(x), generate_Y(x)),
                                              keep_in_memory=True,
                                              load_from_cache_file=False)

    columns = ['input_ids', 'attention_mask', 'labels']
    train_dataset.set_format(type='torch', columns=columns)
    test_dataset.set_format(type='torch', columns=columns)

    # write out the data to disk so we can load it faster later
    train_dataset.save_to_disk("train_data.checkpoint")
    test_dataset.save_to_disk("val_data.checkpoint")
else:
    train_dataset.load_from_disk("train_data.checkpoint")
    test_dataset.load_from_disk("val_data.checkpoint")

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=transformers.TrainingArguments(
        per_device_train_batch_size=MICRO_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=100,
        num_train_epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        fp16=False,
        logging_steps=20,
        evaluation_strategy="steps",
        save_strategy="steps",
        eval_steps=200,
        save_steps=200,
        output_dir="tune",
        save_total_limit=3,
        load_best_model_at_end=True,
        #label_smoothing_factor=0.1,
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
