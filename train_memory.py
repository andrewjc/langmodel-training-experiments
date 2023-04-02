import string

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5TokenizerFast, Adafactor, get_linear_schedule_with_warmup, \
    set_seed
from datasets import load_dataset

from model.AutoTuneMemory import AutoMemoryModule

# Load dataset
dataset = load_dataset("allenai/soda")
train_set, eval_set, test_set = dataset['train'], dataset['validation'], dataset['test']
train_set = train_set.filter(lambda example: all(len(x) > 50 for x in example["dialogue"]), keep_in_memory=True)
eval_set = eval_set.filter(lambda example: all(len(x) > 50 for x in example["dialogue"]), keep_in_memory=True)
test_set = test_set.filter(lambda example: all(len(x) > 50 for x in example["dialogue"]), keep_in_memory=True)


# Load tokenizer and model
tokenizer = T5TokenizerFast.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Hyperparameters
batch_size = 128
max_length = 128
num_epochs = 3
lr = 1e-3
warmup_steps = 500


# Function to tokenize dataset
def tokenize_dataset(dialogue_batch):
    b = dialogue_batch[0]
    inputs = []
    labels = []

    for sentence in b:
        stripped = sentence.translate(str.maketrans('', '', string.punctuation))

        inp = stripped.split()[:-1]
        lbl = stripped.split()[-1]

        inp = " ".join(inp)

        inputs.append(inp)
        labels.append(lbl)

    tokenized_inputs = tokenizer(text=inputs, text_target=labels, return_tensors="pt", padding="max_length", max_length=max_length, truncation=True)

    return tokenized_inputs


# DataLoader
train_loader = DataLoader(train_set, batch_size=1, shuffle=True,
                          collate_fn=lambda batch: tokenize_dataset([item["dialogue"] for item in batch]))

# Optimizer and scheduler
optimizer = Adafactor(model.parameters(), lr=lr, scale_parameter=False, relative_step=False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                            num_training_steps=len(train_loader) * num_epochs)

# Training loop
model.train()


class OrderedMemoryModule:
    def __init__(self):
        self.memory = None

    def __call__(self, *args, **kwargs):
        if 'memory_context' in kwargs:
            memory_context = kwargs['memory_context']
        else:
            memory_context = None

        return memory_context

memoryModule = OrderedMemoryModule()

accumulation_counter = 0
generated_text = ""
for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_idx, inputs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        batch_loss = 0

        memory_context = None
        # Micro-batch loop
        for dialogue_idx in range(inputs["input_ids"].shape[0]):
            input_ids = inputs["input_ids"][dialogue_idx, :].unsqueeze(0)
            attention_mask = inputs["attention_mask"][dialogue_idx, :].unsqueeze(0)
            labels = inputs["labels"][dialogue_idx, :].unsqueeze(0)

            memory_response = memoryModule(input_ids=input_ids, attention_mask=attention_mask, memory_context=memory_context)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            batch_loss += outputs.loss
            accumulation_counter += 1

        if accumulation_counter >= batch_size:
            batch_loss /= accumulation_counter
            batch_loss.backward()
            optimizer.step()
            scheduler.step()
            accumulation_counter = 0
            batch_loss = 0

        # Generate sequences
        set_seed(42)  # Set the seed for reproducibility
        # Input text
        input_text = "Hello how are you"

        # Encode input text
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

        # Generate next tokens
        for i in range(10):
            generated_token_ids = model.generate(input_ids=input_ids, max_length=len(input_ids) + 1, do_sample=True)
            input_ids = torch.cat([input_ids, generated_token_ids[:, -1:]], dim=1)

        # Decode generated tokens
        generated_text = tokenizer.decode(input_ids.tolist()[0], skip_special_tokens=True)

        tqdm.write(f"Batch {batch_idx + 1}/{len(train_loader)} | Batch Loss: {batch_loss} | Generated text: {generated_text}")

    print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {epoch_loss / len(train_loader)}")
