import torch
from torch import nn

from model.AutoTuneMemory import AutoMemoryModule


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, token_dim, hidden_dim, num_layers, max_memory_size, padding_token=0):
        super(LanguageModel, self).__init__()

        self.padding_token = padding_token
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize the AutoMemoryModule
        self.memory_module = AutoMemoryModule(token_dim, hidden_dim, max_memory_size, padding_token)

        # Initialize the encoder
        encoder_layers = []
        for i in range(num_layers):
            encoder_layers.append(nn.GRU(input_size=token_dim, hidden_size=hidden_dim, batch_first=True))
            encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Linear(hidden_dim, token_dim))
            encoder_layers.append(nn.BatchNorm1d(token_dim))
            encoder_layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*encoder_layers)

        # Initialize the decoder
        decoder_layers = []
        for i in range(num_layers):
            decoder_layers.append(nn.Linear(token_dim, hidden_dim))
            decoder_layers.append(nn.BatchNorm1d(hidden_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Linear(hidden_dim, token_dim))
            decoder_layers.append(nn.BatchNorm1d(token_dim))
            decoder_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*decoder_layers)

        # Initialize the output layer
        self.output_layer = nn.Linear(token_dim, vocab_size)

    def forward(self, input_tokens, memory_context):
        # Pass the input tokens and memory context through the AutoMemoryModule
        memory_context, _ = self.memory_module(input_tokens, memory_context)

        # Concatenate the input tokens with the memory context
        combined_input = torch.cat((input_tokens, memory_context), dim=0)

        # Pass the concatenated tokens through the encoder
        encoder_output, _ = self.encoder(combined_input.unsqueeze(0))

        # Pass the encoder output through the decoder
        decoder_output = self.decoder(encoder_output)

        # Compute the logit representing the next predicted token
        output_logit = self.output_layer(decoder_output.squeeze(0))

        return output_logit, memory_context