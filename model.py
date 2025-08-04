import torch
import torch.nn as nn
from config import DEVICE


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        x = self.embedding(x)
        out, h = self.gru(x, h)

        out = out[:, -1, :]

        out = self.fc(out)
        return out, h

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)