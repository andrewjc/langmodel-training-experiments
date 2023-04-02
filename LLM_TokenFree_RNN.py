import numpy as np
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer
import torch_optimizer as optim

import torch
from torch.utils.data import DataLoader
from torch import optim as optim_regular

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
mixed_precision = True

# Training parameters
segment_length = 128
max_train_lines = 1e8
TRAIN_BATCH_SIZE = 32
EVAL_BATCH_SIZE = 256
num_epochs = 10
learning_rate = 0.01
max_sentence_length = 256
temperature = 1.0

data_files = "https://the-eye.eu/public/AI/pile_preliminary_components/github.tar"

tokenizer = LlamaTokenizer.from_pretrained("theblackcat102/llama-fast-tokenizer")
tokenizer.pad_token_id = tokenizer.eos_token_id

def temperature_scaled_init(weight, temperature):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    scale = 1.0 / max(1., float(fan_in) / temperature)
    bound = math.sqrt(scale)
    nn.init.uniform_(weight, -bound, bound)


def init_linear_kaiming_uniform_weights(m):
    if type(m) == nn.Linear:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)


def init_gru_xavier_uniform_weights(m):
    if type(m) == nn.GRU:
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)


def round_to_nearest_power_of_2(n):
    power = round(math.log2(n))
    return 2 ** power


class AutoMemoryModule(nn.Module):
    def __init__(self, embedding_layer, max_sentence_length, max_memory_context, embedding_size, padding_token,
                 device='cpu'):
        super().__init__()
        self.vocab_size = 32000
        self.max_sentence_length = max_sentence_length
        self.max_memory_context = max_memory_context
        self.embedding_size = embedding_size
        self.device = device
        self.padding_token = padding_token
        self.embedding = embedding_layer

        self.score_net = nn.Sequential(
            nn.Linear(self.embedding_size * (self.max_sentence_length), 64),
            nn.ReLU(),
            nn.Linear(64, max_sentence_length),
            nn.Sigmoid(),
        ).to(device=device)

    def forward(self, input_tokens, memory_context):
        if memory_context is None:
            memory_context = torch.zeros(self.max_memory_context, dtype=torch.long).to(device=self.device)
            # fill memory context with padding tokens
            memory_context.fill_(self.padding_token)

        input_tokens = input_tokens.to(device=self.device)
        padded_input_tokens = nn.functional.pad(input_tokens, (0, self.max_sentence_length - input_tokens.shape[-1]),
                                                value=self.padding_token)

        # score the padding_input_tokens
        input_tokens_embedding = self.embedding(padded_input_tokens).to(device=self.device).view(1, -1)
        input_tokens_scoring = self.score_net(input_tokens_embedding).squeeze(dim=0)

        memory_context_embedding = self.embedding(memory_context).to(device=self.device).view(1, -1)
        memory_context_scoring = self.score_net(memory_context_embedding).squeeze(dim=0)

        # filter out the padding tokens from the padded input tokens and their scores

        padding_token_idx = torch.nonzero(padded_input_tokens != self.padding_token).squeeze(dim=1)
        filtered_input_tokens = padded_input_tokens[padding_token_idx]
        filtered_input_tokens_scoring = input_tokens_scoring[padding_token_idx]

        ctx_padding_token_idx = torch.nonzero(memory_context != self.padding_token).squeeze(dim=1)
        filtered_memory_context = memory_context[ctx_padding_token_idx]
        filtered_memory_context_scoring = torch.index_select(memory_context_scoring, 0, ctx_padding_token_idx)

        # combine the filtered input tokens and their scores with the memory context
        # and their scores
        combined_tokens = torch.cat((filtered_input_tokens, filtered_memory_context), dim=0)
        scores = torch.cat((filtered_input_tokens_scoring, filtered_memory_context_scoring), dim=0)

        # remove duplicate tokens and their scores
        unique_tokens, indices = torch.unique(combined_tokens, return_inverse=True)
        unique_scores = torch.full_like(unique_tokens, -1e20, dtype=scores.dtype)
        unique_scores = unique_scores.scatter(0, indices, scores)

        # sort the combined tokens and their scores by the scores
        sorted_scores, sorted_indices = torch.sort(unique_scores, descending=True)
        sorted_combined_tokens = unique_tokens[sorted_indices]

        # trim the combined tokens and their scores to the max memory context size
        trimmed_combined_tokens = sorted_combined_tokens[:self.max_memory_context]
        trimmed_scores = sorted_scores[:self.max_memory_context]

        # pad the trimmed tokens and their scores with padding tokens and -1e20 respectively
        trimmed_combined_tokens = nn.functional.pad(trimmed_combined_tokens,
                                                    (0, self.max_memory_context - trimmed_combined_tokens.shape[-1]),
                                                    value=self.padding_token)
        trimmed_scores = nn.functional.pad(trimmed_scores, (0, self.max_memory_context - trimmed_scores.shape[-1]),
                                           value=-1e20)

        return trimmed_combined_tokens, trimmed_scores


class EncoderBlock(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim, num_layers, dropout):
        super().__init__()
        batch_size, input_seq_len, input_dim = input_shape
        batch_size, output_dim, output_seq_len = output_shape

        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm([output_seq_len, output_dim])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden=None):
        # x has shape (batch_size, input_seq_len, input_dim)
        # hidden has shape (num_layers, batch_size, hidden_dim)
        if hidden is not None:
            h = hidden
        else:
            h = None
        x, h = self.rnn(x, h)
        # x has shape (batch_size, output_seq_len, hidden_dim)
        x = self.linear(x)
        # x has shape (batch_size, output_seq_len, output_dim)
        x = self.norm(x)
        x = self.dropout(x)
        return x, h


class Encoder(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_dim, num_layers, dropout, num_blocks):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_blocks = num_blocks

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_blocks):
            if i == 0:
                input_shape = self.input_shape
            else:
                input_shape = (self.output_shape[0], self.output_shape[2], self.output_shape[1])
            self.encoder_blocks.append(EncoderBlock(input_shape, output_shape, hidden_dim, num_layers, dropout))

    def forward(self, x):
        hidden = None
        for i in range(self.num_blocks):
            x, hidden = self.encoder_blocks[i](x, hidden)
        return x

class DecoderModule(nn.Module):
    def __init__(self, input_shape, vocab_size, hidden_dim, dropout_rate=0.1):
        super(DecoderModule, self).__init__()
        self.batch_size, self.seq_len, self.input_dim = input_shape
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.layer_norm = nn.LayerNorm(self.input_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.ffn = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.vocab_size)
        )

    def forward(self, x):
        # Input shape: (batch_size, seq_len, input_dim)

        # Only consider the last token of the sequence
        x = x[:, -1, :]

        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply dropout
        x = self.dropout(x)

        # Apply feedforward network
        x = self.ffn(x)

        # Output shape: (batch_size, vocab_size)
        return x


class LanguageModel(nn.Module):
    def __init__(self, batch_size, vocab_size, max_sentence_length, embedding_size, layer_ratios, max_memory_size,
                 padding_token=0, temperature=1.0, device='cpu'):
        super(LanguageModel, self).__init__()

        self.padding_token = padding_token
        self.max_sentence_length = max_sentence_length
        self.max_memory_size = max_memory_size
        self.device = device

        self.embedding_size = embedding_size
        self.temperature = temperature

        # Initialize the embedding layer
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        # Initialize the AutoMemoryModule
        self.memory_module = AutoMemoryModule(self.embedding, max_sentence_length, max_memory_size, self.embedding_size,
                                              padding_token, device)

        # Initialize the encoder blocks

        # Create the encoder layers based on the layer sizes
        batch_size = TRAIN_BATCH_SIZE
        input_seq_len = 512
        input_dim = 256
        output_dim = 256
        output_seq_len = 512
        hidden_dim = 256
        num_layers = 2
        dropout = 0.15
        num_blocks = 32
        self.encoder = Encoder((batch_size, input_seq_len, input_dim), (batch_size, output_dim, output_seq_len),
                               hidden_dim,
                               num_layers, dropout, num_blocks)

        # Initialize the decoder
        self.decoder = DecoderModule((batch_size, 256, 256), vocab_size, 256, dropout_rate=0.1)

        self.apply(init_gru_xavier_uniform_weights)
        self.apply(init_linear_kaiming_uniform_weights)

    def positional_encoding(self, position, d_model):
        def cal_angle(position, hid_idx):
            return position / np.power(10000, 2 * (hid_idx // 2) / d_model)

        def get_posi_angle_vec(position):
            return [cal_angle(position, hid_j) for hid_j in range(d_model)]

        sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).to(self.device)

    def forward(self, input_tokens, memory_context, temperature=1.0):

        # Pass the input tokens and memory context through the AutoMemoryModule
        # memory_context, _ = self.memory_module(input_tokens, memory_context)

        # make a new tensor with same batch_size as input_tokens but with max_memory_size as the second dimension
        memory_context = torch.zeros(input_tokens.shape[0], self.max_memory_size, dtype=torch.long,
                                     device=self.device).fill_(self.padding_token)
        memory_context[:, :len(input_tokens)] = input_tokens[:, :len(input_tokens)]

        # Pass input_tokens through the embedding layer
        padded_input_tokens = nn.functional.pad(input_tokens, (0, self.max_sentence_length - input_tokens.shape[-1]),
                                                value=self.padding_token)
        embedded_input_tokens = self.embedding(padded_input_tokens)

        # Pass the memory context through the embedding layer
        embedded_memory_context = self.embedding(memory_context)

        # Get positional encodings
        pos_encodings_input = self.positional_encoding(self.max_sentence_length, self.embedding_size)
        pos_encodings_memory = self.positional_encoding(self.max_memory_size, self.embedding_size)

        # Add positional encodings to the embedded tokens
        embedded_input_tokens = embedded_input_tokens + pos_encodings_input[:padded_input_tokens.shape[1], :]
        embedded_memory_context = embedded_memory_context + pos_encodings_memory

        # Concatenate the input tokens with the memory context
        combined_input = torch.cat((embedded_input_tokens, embedded_memory_context), dim=1)

        # Pass the concatenated tokens through the encoder
        encoder_output = self.encoder(combined_input)

        # Pass the encoder output through the decoder
        decoder_output = encoder_output
        decoder_output = self.decoder(decoder_output)

        # final output probabilities
        output_prob = decoder_output

        return output_prob.squeeze(0), memory_context

    def generate(self, tokenizer, input_ids, max_length=20, temperature=1.0):
        # Set the model to evaluation mode
        self.eval()

        # Initialize the memory context to None
        memory_context = None

        # Convert input_ids to a tensor if necessary
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids).to(self.device)

        # Generate tokens until the maximum length is reached or an end-of-sentence token is generated
        for _ in range(128):

            # Pass the input tokens and memory context through the model

            output_prob, memory_context = self(input_ids, memory_context, temperature)

            log_probs = torch.log_softmax(output_prob, dim=-1)

            next_token = torch.multinomial(torch.exp(log_probs), num_samples=1)

            # Append the next token to the input tensor
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

            # If the next token is an end-of-sentence token, break the loop
            if next_token == tokenizer.eos_token_id:
                break

        # Decode the sequence
        generated_sequence = input_ids.squeeze(0).tolist()
        generated_text = tokenizer.decode(generated_sequence)

        # Set the model back to training mode
        self.train()

        return generated_text


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Training loop
def train():
    from datasets import load_dataset

    # This takes a few minutes to run, so go grab a tea or coffee while you wait :)
    dataset = load_dataset("json", data_files=data_files, split="train", streaming=False)
    dataloader = DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, pin_memory=False)

    scaler = GradScaler(enabled=mixed_precision)

    model = LanguageModel(TRAIN_BATCH_SIZE, 32000, max_sentence_length=max_sentence_length, embedding_size=256,
                          layer_ratios=[0.025, 0.025, 0.015, 0.015, 0.010, 0.010, 0.001, 0.001], max_memory_size=256,
                          padding_token=tokenizer.pad_token_id,
                          temperature=1.0, device=device).to(device)
    print(model)

    #model.load_state_dict(torch.load(f"/content/drive2/MyDrive/model_3330_6.0924072265625.pt"))

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Lamb(model.parameters(), lr=0.01)

    criterion = nn.CrossEntropyLoss()

    # Add super convergence techniques
    loops = 50
    scheduler = optim_regular.lr_scheduler.CosineAnnealingLR(optimizer, T_max=loops, eta_min=1e-6, last_epoch=-1)


    progress_bar = tqdm(desc="Training", position=0, leave=True)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(dataloader):

            X_batch = batch["text"]

            input_batch = []
            target_batch = []

            batch_utf8_encoded = [list(x.encode('utf-8')) for x in X_batch]

            # Find the maximum length of the encoded strings in the batch
            max_len = max(len(x) for x in batch_utf8_encoded)

            # Pad the encoded strings with zeros to make them all the same length
            batch_padded = [x + [0] * (max_len - len(x)) for x in batch_utf8_encoded]

            # Convert the padded batch to a NumPy array
            batch_array = np.array(batch_padded)

            for segment in batch_array:
                if len(segment) > max_sentence_length + 1:
                    segment = segment[0: max_sentence_length + 1]

                    X_data = segment[:-1]
                    y_data = segment[-1]

                    X_data = torch.tensor(X_data, dtype=torch.long).to(device)
                    y_data = torch.tensor(y_data, dtype=torch.long).to(device)

                    input_batch.append(X_data)
                    y_onehot = F.one_hot(y_data, num_classes=32000).float()
                    target_batch.append(y_onehot)

            # convert from list of tensors to single tensor
            input_batch = torch.vstack(input_batch).to(device)
            target_batch = torch.vstack(target_batch).to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs, memory_context = model(input_batch, memory_context=None, temperature=temperature)
                loss = criterion(outputs, target_batch.squeeze(-1))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            progress_bar.set_description(
                f"Batch ID: {batch_idx} - Loss: {loss.item():.4f} LR: {optimizer.param_groups[0]['lr']:.6f}")

            if batch_idx % 25 == 0:
                prompt = "Hello how are you?"
                input_data = tokenizer.encode(prompt, return_tensors="pt").to(device)
                output = model.generate(tokenizer, input_ids=input_data, max_length=max_sentence_length)
                print(f"Generated sentence: {output}")

            if batch_idx % 50 == 0:
                torch.save(model.state_dict(), f"/content/drive2/MyDrive/model_{batch_idx}_{loss}.pt")

            scheduler.step(loss)


if __name__ == '__main__':
    train()