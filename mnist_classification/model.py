import torch
import torch.nn as nn

class ImgClassifier(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size):
        self.input_size = input_size
        self.output_size = output_size

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, 500),
            nn.LeakyReLU(),
            nn.BatchNorm1d(500),
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),
            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, 25),
            nn.LeakyReLU(),
            nn.BatchNorm1d(25),
            nn.Linear(25,output_size),
            nn.LogSoftmax(dim=-1),
        )
        
    def forward(self,x):
        # |x| = (bs,input_size)
        
        y = self.layers(x) # |y| = (bs,output_size)
        
        return y
    