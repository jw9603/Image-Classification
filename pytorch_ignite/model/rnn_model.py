import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size, n_layers=4, dropout_p=.3):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        
        super().__init__()
        
        # References: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_p,
            bidirectional=True
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.LogSoftmax(dim=-1)
        )
    
    def forward(self, x):
        # |x| = (bs, h, w), h 가 LSTM에 들어갈 때는 time-step이라 생각하면 됨
        z, _ = self.rnn(x)
        # |z| = (bs, h, hs * 2), bidirection이니까 hs * 2
        z = z[:, -1]
        y = self.layers(z)
        # |y| = (bs, output_size)
        return y