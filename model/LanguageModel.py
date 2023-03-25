import torch
from torch import nn

from model.AutoTuneMemory import AutoMemoryModule


class RNNEncoderBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RNNEncoderBlock, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.bn(out[:, -1, :])
        out = self.relu(out)
        return out

class MemoryEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size):
        super(MemoryEncoder, self).__init__()
        self.rnn_blocks = nn.Sequential(
            RNNEncoderBlock(input_size, hidden_size),
            RNNEncoderBlock(hidden_size, hidden_size),
            RNNEncoderBlock(hidden_size, hidden_size),
            RNNEncoderBlock(hidden_size, hidden_size)
        )
        self.linear = nn.Linear(hidden_size, memory_size)
        self.concat_bn = nn.BatchNorm1d(input_size + memory_size)
        self.fc_blocks = nn.Sequential(
            nn.Linear(input_size + memory_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, input_size)
        )
        
    def forward(self, x, memory_context):
        # Pass input_tokens through RNN encoder blocks
        out = self.rnn_blocks(x)
        out = self.linear(out)
        # Combine memory_context and RNN output
        out = torch.cat([out, memory_context], dim=1)
        out = self.concat_bn(out)
        # Pass through fully connected layers
        out = self.fc_blocks(out)
        return out

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, input_size)
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        # Embed input token
        x = self.embedding(x)
        # Pass through GRU layer
        out, hidden = self.gru(x, hidden)
        # Pass through linear layer and return output
        out = self.linear(out[:, -1, :])
        return out, hidden

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, token_dim, hidden_dim, num_layers, max_memory_size, padding_token=0):
        super(LanguageModel, self).__init__()

        self.padding_token = padding_token
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initialize the AutoMemoryModule
        self.memory_module = AutoMemoryModule(token_dim, hidden_dim, max_memory_size, padding_token)

        self.memory_encoder = MemoryEncoder(token_dim, hidden_dim, max_memory_size)

        self.decoder = Decoder(token_dim, hidden_dim, vocab_size)

    def forward(self, input_tokens, memory_context):
        # Pass the input tokens and memory context through the AutoMemoryModule
        memory_context, _ = self.memory_module(input_tokens, memory_context)

        # Pass the input tokens and memory context through the MemoryEncoder
        encoded_context = self.memory_encoder(input_tokens, memory_context)

        # Pass the encoder output through the decoder
        decoder_output = self.decoder(encoded_context)

        return decoder_output, memory_context