from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Trainer():
    
    def __init__(self,model,optimizer,crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit
        
        super().__init__()
        
    
    def train_bs(self,x,y,config):
        self.model.train()
        
        # shuffle before begins.
        indices = torch.randperm(x.size(0),device=x.device)
        x = torch.index_select(x,dim=0,index=indices).split(config.batch_size,dim=0)
        y = torch.index_select(y,dim=0,index=indices).split(config.batch_size,dim=0)
        # # x and y are tuples, each containing tensors that are split by batch size.
        
        total_loss = 0
        
        for i, (x_i,y_i) in enumerate(zip(x,y)):
            # This single loop iteration processes a batch of data equal to the specified batch size, representing one iteration.

            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i,y_i)
            
            self.optimizer.zero_grad()
            loss_i.backward()
            
            self.optimizer.step()
            
            if config.verbose >= 2:
                print(f"Train Iteration({i + 1}/{len(x)}): loss={float(loss_i):.4e}")
            
            total_loss += float(loss_i)
            
        # Return the average loss per batch
        return total_loss / len(x) 

    
    def validate_bs(self,x,y,config):
        self.model.eval()
        
        with torch.no_grad():
            
            indices = torch.randperm(x.size(0),device=x.device)
            x = torch.index_select(x,dim=0,index=indices).split(config.batch_size,dim=0)
            y = torch.index_select(y,dim=0,index=indices).split(config.batch_size,dim=0)
            
            total_loss = 0
            
            for i,(x_i,y_i) in enumerate(zip(x,y)):
                # This single loop iteration processes a batch of data equal to the specified batch size, representing one iteration.
                
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i,y_i)
                
                if config.verbose >= 2:
                    print(f"Valid Iteration({i + 1}/{len(x)}): loss={float(loss_i):.4e}")
                    
                total_loss += float(loss_i)
            
            # Return the average loss per batch
            return total_loss / len(x)
        
    
    def train(self,train_data,valid_data,config):
        
        lowest_loss = np.inf
        best_model = None
        
        for epoch in range(config.n_epochs):
            
            train_loss = self.train_bs(train_data[0],train_data[1],config)
            valid_loss = self.validate_bs(valid_data[0],valid_data[1],config)
            
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())
                
#                 print(f'{epoch+1}-epoch의 model의 state_dict: {self.model.state_dict()}')
                
            print(f"Epoch({epoch + 1}/{config.n_epochs}): train_loss={train_loss:.4e}  valid_loss={valid_loss:.4e}  lowest_loss={lowest_loss:.4e}")
            
        
        print(f'최종 모델 호출')
        self.model.load_state_dict(best_model)