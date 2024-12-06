import argparse
import torch
import torch.nn
import sys
from model import ImageClassifier
from data_loader import load_mnist
import os


# './result_model/' 디렉터리에 있는 모든 파일을 가져오는 함수
def get_model_list(directory='./result_model/'):
    # 디렉터리에 있는 모든 파일 이름을 리스트로 반환
    if os.path.exists(directory) and os.path.isdir(directory):
        return [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]
    else:
        print(f"Directory '{directory}' does not exist or is not a directory.")
        return []
    
def load(fn,device):
    d = torch.load(fn, map_location=device)
    return d['model']


def test(model, x, y):
    
    model.eval()
    with torch.no_grad():
        y_hat = model(x)
        
        correct_cnt = (y.squeeze() == torch.argmax(y_hat,dim=-1)).sum()
        total_cnt = x.size(0)
        
        acc = correct_cnt / total_cnt
    
    return acc    
    
    
    pass

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model_list = get_model_list('./result_model/')
    
    x, y = load_mnist(is_train=False, flatten=True)
    
    x, y = x.to(device), y.to(device)    
    
    
    model = ImageClassifier(28**2, 10).to(device)
    accuracy = []
    for i in model_list:
        model.load_state_dict(load(fn=i, device=device))
        acc = test(model=model, x=x, y=y)
        accuracy.append((i,acc))
        
    print('\n'.join([f"{i}: {j:.4f}" for i, j in accuracy]))
