import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import numpy as np
from dataset import WINDOW_SIZE
import time




class WIFI_MODEL(torch.nn.Module):
    def __init__(self, input_dim=229, hidden_dim=229, output_dim=3, n_layers=4, dropout=0.2, device='cuda:0'):
        super(WIFI_MODEL, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device

        self.linear_in = torch.nn.Linear(input_dim, hidden_dim)
        self.linear_in2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_in3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear_in4 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.lstm_wifi = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=n_layers)

        self.lstm_overall = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=n_layers, dropout=dropout)

        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x, hidden):

        x = self.dropout(x)
        x = self.linear_in(x)
        x = F.relu(x)
        x = self.linear_in2(x)
        x = F.relu(x)
        x = self.linear_in3(x)
        x = F.relu(x)
        x = self.linear_in4(x)
        x = F.relu(x)

        output, hidden = self.lstm_overall(x, hidden)

        output = self.fc(x)
        return output, hidden

    def init_hidden(self, bs):
        hidden_cell = (torch.zeros(self.n_layers, 1, self.hidden_dim, device=self.device),
                       torch.zeros(self.n_layers, 1, self.hidden_dim, device=self.device))
        return hidden_cell

    def init_hidden_2(self, bs):
        hidden_cell = (torch.zeros(self.n_layers, 1, self.hidden_dim, device=self.device),
                       torch.zeros(self.n_layers, 1, self.hidden_dim, device=self.device))
        return hidden_cell



