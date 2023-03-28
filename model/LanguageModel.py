import math

import torch
from torch import nn

from model.AutoTuneMemory import AutoMemoryModule

import torch
import torch.nn as nn
import torch.nn.functional as F


def temperature_scaled_init(weight, temperature):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    scale = 1.0 / max(1., float(fan_in) / temperature)
    bound = math.sqrt(scale)
    nn.init.uniform_(weight, -bound, bound)


def gumbel_softmax(logits, temperature, hard=False, dim=-1):
    gumbels = -torch.empty_like(logits).exponential_().log()
    logits_with_gumbel_noise = (logits + gumbels) / temperature
    y_soft = F.softmax(logits_with_gumbel_noise, dim=dim)

    if hard:
        _, max_index = y_soft.max(dim, keepdim=True)
        y_hard = torch.zeros_like(logits).scatter_(dim, max_index, 1.0)
        y_hard = y_hard - y_soft.detach() + y_soft
    else:
        y_hard = y_soft

    return y_hard


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


class EncoderBlock(nn.Module):
    def __init__(self, input_size, gru_in, gru_out, output_size, dropout=0.2, temperature=1.0):
        super(EncoderBlock, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.temperature = temperature

        # Add a dimensionality reduction layer
        self.linear_dim_red = nn.Linear(input_size, gru_in)
        temperature_scaled_init(self.linear_dim_red.weight, temperature)
        self.activation_dim_red = nn.ReLU()

        # Initialize a GRU-based recurrent block
        self.rnn = nn.GRU(gru_in, gru_out, batch_first=True, bidirectional=True)
        self.ln_rnn = nn.LayerNorm(gru_out*2)
        self.dropout = nn.Dropout(p=dropout*2)

        # Initialize a dense layer
        self.linear = nn.Linear(gru_out*2, output_size)
        temperature_scaled_init(self.linear.weight, temperature)

        # Add layer normalization to the recurrent block and dense layer
        self.ln_linear = nn.LayerNorm(output_size)

        # Add dropout after the recurrent block and dense layer

        self.activation = nn.ReLU()

        self.apply(init_gru_xavier_uniform_weights)
        self.apply(init_linear_kaiming_uniform_weights)

    def forward(self, x, h=None):

        # Apply dimensionality reduction
        x = self.linear_dim_red(x)
        x = self.activation_dim_red(x)

        # Pass the input and previous hidden state through the recurrent block
        out, h = self.rnn(x, h)

        # Apply layer normalization and dropout to the output of the recurrent block
        out = self.ln_rnn(out)
        out = self.dropout(out)

        # Pass the output of the recurrent block through a dense layer
        out = self.ln_linear(self.linear(out))

        out = self.activation(out)
        out = self.dropout(out)

        return out, h


class DecoderLayer(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, max_sentence_length, hidden_dim, layer_ratios, max_memory_size,
                 padding_token=0, temperature=1.0, device='cpu'):
        super(LanguageModel, self).__init__()

        self.padding_token = padding_token
        self.max_sentence_length = max_sentence_length
        self.hidden_dim = hidden_dim
        self.device = device

        self.embedding_size = 256
        self.temperature = temperature

        # Initialize the embedding layer
        self.embedding = nn.Embedding(vocab_size, self.embedding_size)

        # Initialize the AutoMemoryModule
        self.memory_module = AutoMemoryModule(self.embedding, max_sentence_length, max_memory_size, self.embedding_size,
                                              padding_token, device)

        # Initialize the encoder blocks
        first_encoder_input_size = (self.embedding_size * max_sentence_length) + (self.embedding_size * max_memory_size)
        last_encoder_output_size = 128

        encoder_input_sizes = [first_encoder_input_size] + [
            int(last_encoder_output_size + (first_encoder_input_size - last_encoder_output_size) * ratio) for ratio in
            layer_ratios]
        encoder_output_sizes = encoder_input_sizes[1:] + [last_encoder_output_size]

        # Create the encoder layers based on the layer sizes
        encoder_layers = []
        for i in range(len(encoder_input_sizes)):
            input_size = round_to_nearest_power_of_2(encoder_input_sizes[i])
            output_size = round_to_nearest_power_of_2(encoder_output_sizes[i])
            layer = EncoderBlock(input_size, 256, 256, output_size, dropout=0.15, temperature=temperature)
            encoder_layers.append(layer)

        self.encoder = nn.ModuleList(encoder_layers)

        # Initialize the decoder
        self.decoder = DecoderLayer(last_encoder_output_size, [128, 128], vocab_size)

        self.apply(init_gru_xavier_uniform_weights)
        self.apply(init_linear_kaiming_uniform_weights)

    def forward(self, input_tokens, memory_context, temperature=1.0):

        # Pass the input tokens and memory context through the AutoMemoryModule
        memory_context, _ = self.memory_module(input_tokens, memory_context)

        # Pass input_tokens through the embedding layer
        padded_input_tokens = nn.functional.pad(input_tokens, (0, self.max_sentence_length - input_tokens.shape[-1]),
                                                value=self.padding_token)
        embedded_input_tokens = self.embedding(padded_input_tokens)

        # Pass the memory context through the embedding layer
        embedded_memory_context = self.embedding(memory_context)

        # Concatenate the input tokens with the memory context
        combined_input = torch.cat((embedded_input_tokens, embedded_memory_context), dim=0)

        # Reshape the combined input to be of shape (1, max_memory_size + token_dim)
        combined_input = combined_input.reshape(1, -1)

        # Pass the concatenated tokens through the encoder
        encoder_output = combined_input
        for i in range(len(self.encoder)):
            encoder_output, _ = self.encoder[i](encoder_output)

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

        # Initialize the generated token list with the input_ids
        generated_tokens = input_ids.tolist()
        # Generate tokens until the maximum length is reached or an end-of-sentence token is generated
        while len(generated_tokens[0]) < max_length:
            # Convert the generated tokens to a tensor
            input_ids = torch.tensor(generated_tokens).to(self.device).squeeze()

            # Pass the input tokens and memory context through the model
            output_prob, memory_context = self(input_ids, memory_context, temperature)

            # Sample the next token from the output probabilities

            next_token = torch.multinomial(output_prob.exp(), num_samples=1).item()

            # Add the next token to the generated token list
            generated_tokens[0].append(next_token)

            # If the next token is an end-of-sentence token, break the loop
            if next_token == self.padding_token:
                break


        # Decode the sequence
        generated_tokens = [tokenizer.decode(token) for token in generated_tokens]

        # Set the model back to training mode
        self.train()

        return generated_tokens
