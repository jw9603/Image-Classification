import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from trainer import Trainer

from data_loader import get_loaders
from model.fc_model import FcClassifier
from model.cnn_model import ConvolutionalClassifier


def define_argparser():
    p = argparse.ArgumentParser()
    
    p.add_argument('--model_file', required=True)
    p.add_argument('--model', type=str, default='fc')
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=30)
    p.add_argument('--verbose', type=int, default=2)
    
    config = p.parse_args()
    
    return config

def model_choose(config):
    if config.model == 'fc':
        model = FcClassifier(28**2, 10)
    elif config.model == 'cnn':
        model = ConvolutionalClassifier(10)
    else:
        raise NotImplementedError('You need to tell me which model to use.')
    return model


def main(config):
    
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device(f"cuda:{config.gpu_id}")
    
    train_loader, valid_loader, test_loader = get_loaders(config)
    
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Valid: {len(valid_loader.dataset)}")
    print(f"Test: {len(test_loader.dataset)}")
    
    # model = FcClassifier(28**2, 10).to(device) # Before
    model = model_choose(config).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()
    
    if config.verbose >= 2:
        print(f"model: {model}")
        print(f"optimizer: {optimizer}")
        print(f"crit: {crit}")    
    trainer = Trainer(config)
    best_model = trainer.train(model=model, crit=crit, optimizer=optimizer, train_loader=train_loader, valid_loader=valid_loader)
    
    print(f"best_model",best_model)
    
    
if __name__ == '__main__':
    config = define_argparser()
    main(config)
