from transformers import AutoModelForSeq2SeqLM, LlamaTokenizer

from model.LanguageModel import LanguageModel

tokenizer = LlamaTokenizer.from_pretrained("theblackcat102/llama-fast-tokenizer")

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from typing import Tuple


# Custom Dataset class
class LargeJsonlDataset(Dataset):
    def __init__(self, file_path: str, segment_length: int = 48, max_train_lines: int = 1e5):
        self.row_count = 0
        self.file_path = file_path
        self.segment_length = segment_length
        self.line_offsets = []

        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            offset = 0
            for line in file:
                if max_train_lines is not None:
                    if self.row_count >= max_train_lines:
                        print("Got enough lines!")
                        break
                self.row_count += 1
                self.line_offsets.append(offset)
                offset += len(line.encode())  # Get the byte length of the line

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as file:
            file.seek(self.line_offsets[index])
            line = file.readline()
            data = json.loads(line)
            sentence = data["text"]
            segments = [sentence[i:i + self.segment_length] for i in range(0, len(sentence), self.segment_length)]
            return segments


# Training parameters
file_path = f"E:\datasets\Literotica.jsonl"
segment_length = 48
max_train_lines = 1e5
batch_size = 16
num_epochs = 10
learning_rate = 0.001
max_sentence_length = 256


def custom_collate_fn(batch):
    batch_segments = []
    for segments in batch:
        batch_segments.append(segments)
    return batch_segments


# Prepare dataset and dataloader
dataset = LargeJsonlDataset(file_path, segment_length, max_train_lines)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)

# Initialize your language model, criterion, and optimizer here.
model = LanguageModel(32000, max_sentence_length, 512, 2, 512, tokenizer.pad_token_id)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Training loop
def process_segment(segment: str, max_length: int = 256, padding_token: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    tokens = tokenizer.encode(segment, max_length=max_length+1, padding="max_length", truncation=True)

    input_data = torch.tensor(tokens[:-1], dtype=torch.long)
    target_data = torch.tensor(tokens[1:], dtype=torch.long)

    return input_data, target_data


for epoch in range(num_epochs):
    for batch_idx, batch_segments in enumerate(dataloader):
        optimizer.zero_grad()

        batch_loss = 0
        for segments in batch_segments:
            memory_context = None
            for segment in segments:
                input_data, target_data = process_segment(segment, max_sentence_length)
                output, memory_context = model(input_data, memory_context)
                loss = criterion(output, target_data)
                batch_loss += loss

        batch_loss /= len(batch_segments)
        batch_loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {batch_loss.item()}")
