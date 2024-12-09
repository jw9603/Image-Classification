import torch
import torch.nn as nn

class ConvolutionBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        super().__init__()
        
        
        # References: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, (3,3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        # |x| = (bs, in_channels, h, w)
        return self.layers(x) # (bs, out_channels, h, w)
    
class ConvolutionalClassifier(nn.Module):
    def __init__(self,output_size):
        self.output_size = output_size

        super().__init__()
        
        # |x| = (bs, 1, 28, 28)
        self.cnn_block = nn.Sequential(
            ConvolutionBlock(1, 32),
            ConvolutionBlock(32, 64),
            ConvolutionBlock(64, 128),
            ConvolutionBlock(128, 256),
            ConvolutionBlock(256, 512)
        )
        
        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.LogSoftmax(dim=-1)
        )
    def forward(self, x):
        if x.dim() == 2:
            x = x.view(-1, int(x.size(-1)**0.5), int(x.size(-1)**0.5))
        
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))
        # |x| = (bs,1, h, w)
        
        z = self.cnn_block(x) 
        # |z| = (bs, 512, 1, 1)
        
        y = self.layers(z.squeeze())
        # |y| = (bs, output_size)
        
        return y
        