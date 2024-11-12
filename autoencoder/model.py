import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    
    def __init__(self, input_size=28*28, btl_size=2,layers=7):
        super().__init__()
        
        self.btl_size = btl_size
        
        # Layer sizes 등차 수열 계산
        encoder_layer_sizes = [input_size - i * (input_size - btl_size) // (layers - 1) for i in range(layers)]
        decoder_layer_sizes = list(reversed(encoder_layer_sizes))
        
        # Encoder 정의
        encoder_layers = []
        for i in range(len(encoder_layer_sizes) - 1):
            encoder_layers.append(nn.Linear(encoder_layer_sizes[i], encoder_layer_sizes[i + 1]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.BatchNorm1d(encoder_layer_sizes[i + 1]))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder 정의
        decoder_layers = []
        for i in range(len(decoder_layer_sizes) - 1):
            decoder_layers.append(nn.Linear(decoder_layer_sizes[i], decoder_layer_sizes[i + 1]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.BatchNorm1d(decoder_layer_sizes[i + 1]))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


