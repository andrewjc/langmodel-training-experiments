import torch
import torch.nn as nn

class LLMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(LLMBlock, self).__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, input_size)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(hidden_size)

    def forward(self, input, hidden=None):
        output, hidden = self.gru(input, hidden)
        output = self.dropout(output)
        output = self.activation(self.batch_norm(self.linear1(output)))
        output = self.linear2(output)

        return output, hidden
