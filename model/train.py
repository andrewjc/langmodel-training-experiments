import concurrent

from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, LlamaTokenizer
from torch.cuda.amp import autocast, GradScaler
from model.LanguageModel import LanguageModel
from model.swa import SWA

tokenizer = LlamaTokenizer.from_pretrained("theblackcat102/llama-fast-tokenizer")

import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from typing import Tuple
from filelock import FileLock


def process_line(line, tokenizer, segment_length, output_file_path):
    try:
        data = json.loads(line)
        sentence = data["text"]
        segments = [sentence[i:i + segment_length] for i in range(0, len(sentence), segment_length)]
        tokenized_segments = [tokenizer.encode(segment, return_tensors="pt").tolist() for segment in segments]
        with FileLock(output_file_path + ".lock"):
            with open(output_file_path, "a", encoding="utf-8", errors="ignore") as out_file:
                out_file.write(json.dumps(tokenized_segments) + "\n")
        return True
    except Exception as e:
        print(f"Error processing line: {line}")
        print(f"Exception: {e}")
        return False


def pretokenize_and_save(input_file_path: str, output_file_path: str, segment_length: int = 48, num_threads: int = 4):


    with open(input_file_path, "r", encoding="utf-8", errors="ignore") as in_file, \
         concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:

        futures = []
        for line in in_file:
            future = executor.submit(process_line, line, tokenizer, segment_length, output_file_path)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if not result:
                print("An error occurred while processing a line.")

    print("Done!")

def pretokenize_and_saveOld(input_file_path: str, output_file_path: str, segment_length: int = 48):
    with open(input_file_path, "r", encoding="utf-8", errors="ignore") as in_file, \
         open(output_file_path, "w", encoding="utf-8", errors="ignore") as out_file:
        for line in in_file:
            data = json.loads(line)
            sentence = data["text"]
            segments = [sentence[i:i + segment_length] for i in range(0, len(sentence), segment_length)]
            tokenized_segments = [tokenizer.encode(segment, return_tensors="pt").tolist() for segment in segments]
            out_file.write(json.dumps(tokenized_segments) + "\n")

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
                offset += len(line.encode()) +1  # Get the byte length of the line

    def __len__(self):
        return len(self.line_offsets)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        with open(self.file_path, "r", encoding="utf-8", errors="ignore") as file:
            file.seek(self.line_offsets[index])
            line = file.readline()
            data = json.loads(line)
            return data


# Training parameters
file_path = f"src.jsonl"
tokenized_file_path = f"/home/andrewc/Development/literotica_tokenized.jsonl"
segment_length = 128
max_train_lines = 1e8
TRAIN_BATCH_SIZE = 256
EVAL_BATCH_SIZE = 256
num_epochs = 10
learning_rate = 0.001
max_sentence_length = 256
temperature = 1.0


def custom_collate_fn(batch):
    batch_segments = []
    for segments in batch:
        batch_segments.append(segments)
    return batch_segments

def process_segment(tokens: str, max_length: int = 256, padding_token: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    input_data = torch.tensor(tokens[:-1], dtype=torch.long)
    target_data = torch.tensor(tokens[-1], dtype=torch.long)
    return input_data, target_data



# Training loop


def train():
    # Prepare dataset and dataloader

    # TODO: PRETOKENIZE THE JSONL FILE AND SAVE TO THE TOKENIZE FILE PATH
    #pretokenize_and_save(file_path, tokenized_file_path, segment_length, 15)

    dataset = LargeJsonlDataset(tokenized_file_path, segment_length, max_train_lines)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, num_workers=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    scaler = GradScaler(enabled=False)

    model = LanguageModel(32000, max_sentence_length=max_sentence_length, hidden_dim=64, layer_ratios=[0.015, 0.015, 0.015, 0.015, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005], max_memory_size=256, padding_token=tokenizer.pad_token_id,
                          temperature=1.0, device=device).to(device)
    print(model)

    base_opt = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    optimizer = SWA(base_opt, swa_start = 10, swa_freq = 5)

    #optimizer = base_opt

    # Add super convergence techniques
    loops = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=loops, eta_min=1e-6, last_epoch=-1)

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        batch_loss = 0
        segment_batch_count = 0
        optimizer.zero_grad()
        progress_bar = tqdm(total=len(dataloader), desc="Training", position=0, leave=True)

        for batch_idx, batch_segments in enumerate(dataloader):
            if batch_segments is None:
                continue

            if batch_segments[0] is None:
                continue

            if len(batch_segments[0]) == 0:
                continue

            batch_segments = batch_segments[0]

            memory_context = None

            for segment in batch_segments:
                segment = segment[0]
                #print("Segment length: ", len(segment))
                #print("Segment: ", segment)

                try:

                    input_data, target_data = process_segment(segment, max_sentence_length)
                    input_data = input_data.to(device)
                    target_data = target_data.to(device)

                    def calcLoss(output, target):
                        return torch.nn.functional.cross_entropy(output, target, ignore_index=tokenizer.pad_token_id)

                    with autocast():
                        output, memory_context = model(input_data, memory_context)
                        output = output.view(1, -1)
                        target = target_data.unsqueeze(0)
                        loss = calcLoss(output, target)

                    batch_loss += loss
                    segment_batch_count += 1

                    if (segment_batch_count + 1) % TRAIN_BATCH_SIZE == 0:
                        batch_loss /= TRAIN_BATCH_SIZE
                        batch_loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()  # Update the scheduler after optimizer.step()
                        progress_bar.set_description(
                            f"Epoch {epoch + 1}/{num_epochs} - Loss: {batch_loss:.4f} LR: {scheduler.get_last_lr()[0]:.6f}")
                        batch_loss = 0

                    # Check if we should run the generate sequence test code
                    if (segment_batch_count + 1) % EVAL_BATCH_SIZE == 0:
                        prompt = "Hello how are you?"
                        input_data = tokenizer.encode(prompt, return_tensors="pt").to(device)
                        output = model.generate(tokenizer, input_ids=input_data, max_length=max_sentence_length)
                        print(f"Generated sentence: {output}")

                    if (segment_batch_count + 1) % 100000 == 0:
                        optimizer.update_swa()
                        optimizer.swap_swa_sgd()
                        torch.save(model.state_dict(), f"model_{segment_batch_count}.pt")
                        optimizer.swap_swa_sgd()

                except Exception as e:
                    print("Error: ", e)
                    # clear gradients
                    optimizer.zero_grad()
                    batch_loss = 0
                    continue

            progress_bar.update(1)

    # Return the final model and training statistics
    return model


if __name__ == '__main__':
    train()