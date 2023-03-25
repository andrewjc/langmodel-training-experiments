import torch
import torch.nn as nn

from model.AutoTuneMemory import AutoMemoryModule
from model.LMBlock import LLMBlock


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, max_sentence_length, hidden_size, num_layers=1, dropout=0.0):
        super(LanguageModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.auto_memory = AutoMemoryModule(max_sentence_length, hidden_size)

        self.encoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.encoder_blocks.append(LLMBlock(hidden_size, hidden_size, dropout=dropout))

        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers):
            self.decoder_blocks.append(LLMBlock(hidden_size, hidden_size, dropout=dropout))

        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_tokens, memory_context):
        # Embed input tokens
        embedded = self.embedding(input_tokens)

        # Concatenate memory context
        embedded = torch.cat([embedded, memory_context.unsqueeze(1)], dim=1)

        # Pass through auto memory
        memory_context, importance_scores = self.auto_memory(embedded)

        # Pass through encoder layer blocks
        hidden = None
        for block in self.encoder_blocks:
            embedded, hidden = block(embedded, hidden)

        # Pass through decoder layer blocks
        for block in self.decoder_blocks:
            embedded, _ = block(embedded, hidden)

        # Pass through output layer
        output = self.output_layer(embedded)

        return output[:, -1, :], memory_context, importance_scores
