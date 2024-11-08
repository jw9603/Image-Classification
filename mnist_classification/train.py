import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from model import ImgClassifier
from trainer import Trainer
from utils import load_mnist

def define_argparse():
    
    p = argparse.ArgumentParser(description="Train a model with specific parameters.")
    
    
    # By default, 'required' is set to False, making the argument optional.
    # Setting 'required=True' makes the argument mandatory.
    p.add_argument('--model',required=True)
    p.add_argument('--gpu_id',type=int,default=0 if torch.cuda.is_available() else -1)
    
    p.add_argument('--train_ratio',type=float,default=0.8)
    p.add_argument('--batch_size',type=int,default=64)
    p.add_argument('--n_epochs',type=int,default=30)
    p.add_argument('--verbose',type=int,default=2)
    
    config = p.parse_args()
    
    return config

def main(config):
    
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f"cuda:{config.gpu_id}")
    
    # Load Dataset
    x, y = load_mnist(is_train=True,flatten=True)
    # |x| = (bs,input_size) = (60000,784) 
    # |y| = (bs,) = (60000)
    
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt
    
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices,
    ).to(device).split([train_cnt,valid_cnt],dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices,
    ).to(device).split([train_cnt,valid_cnt],dim=0)
    
    print("Train:", x[0].shape, y[0].shape) # torch.Size([48000, 784]) torch.Size([48000])
    print("Valid:", x[1].shape, y[1].shape) # torch.Size([12000, 784]) torch.Size([12000])
    
    # MODEL
    model = ImgClassifier(input_size=x[0].size(1),output_size=int(max(y[0]))+1)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()
    
    # TRAIN
    trainer = Trainer(model=model,optimizer=optimizer,crit=crit)
    
    trainer.train(train_data=(x[0],y[0]),valid_data=(x[1],y[1]),config=config)
    
    # Save the best model
    torch.save({
        'model':trainer.model.state_dict(),
        'config':config,
    },config.model)
    
    
if __name__ == '__main__':
    config = define_argparse()
    main(config)