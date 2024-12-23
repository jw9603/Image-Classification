import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils as torch_utils
from copy import deepcopy
from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar


VERBOSE_silent = 0
VERBOSE_epoch_wise = 1
VERBOSE_batch_wise = 2

class MyEngine(Engine):
    # what is Engine class : Runs a given process_function over each batch of a dataset, emitting events as it goes.
    def __init__(self,func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config
        
        super().__init__(func) # process function
        
        self.best_loss = np.inf
        self.best_model = None
        self.device = next(model.parameters()).device
        
    @staticmethod
    def train(engine, mini_batch): # engine은 현재 실행 중인 Engine 객체
        
        engine.model.train()
        
        engine.optimizer.zero_grad()
        
        x, y = mini_batch
        x, y = x.to(engine.device), y.to(engine.device)
        
        # Model 
        y_hat = engine.model(x)
        
        loss = engine.crit(y_hat, y)
        loss.backward()
        
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            acc = (torch.argmax(y_hat,dim=-1) == y).sum() / float(y.size(0))
        else:
            acc = 0
        # References: https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
        if engine.config.max_grad > 0:
            torch_utils.clip_grad_norm_(
                engine.model.parameters(),
                engine.config.max_grad,
                norm_type=2
            )
        
        engine.optimizer.step()
        
        return {
            'loss': float(loss),
            'accuracy': float(acc)
        }
    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()
        
        with torch.no_grad():
            x, y = mini_batch
            x, y = x.to(engine.device), y.to(engine.device)
            
            # Model
            y_hat = engine.model(x)
            
            loss = engine.crit(y_hat, y)
            
            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                acc = (torch.argmax(y_hat,dim=-1) == y).sum() / y.size(0)
            else:
                acc = 0
        
        return {
            'loss': float(loss),
            'accuracy': float(acc)
        }
        
    @staticmethod
    def attach(train_engine, validate_engine, verbose=VERBOSE_batch_wise):
        
        def attach_running_average(engine,metric_name):
            RunningAverage(output_transform=lambda x:x[metric_name]).attach(engine, metric_name)
        
        training_metrics = ['loss', 'accuracy']
        
        for metric in training_metrics:
            attach_running_average(engine=train_engine, metric_name=metric)
        
        
        if verbose >= VERBOSE_batch_wise:
            pbar = ProgressBar(bar_format=None,ncols=150)
            pbar.attach(engine=train_engine, metric_names=training_metrics) # pbar이 iteration마다 나올것임: 1iteration은  batch_size만큼 학습을 의미함
        
        if verbose >= VERBOSE_epoch_wise:
            @train_engine.on(Events.EPOCH_COMPLETED) # 학습 데이터의 훈련이 1epoch 끝날때마다
            def print_train_logs(engine):
                print(f"Epoch: {engine.state.epoch} - loss={engine.state.metrics['loss']:.4e} accuracy={engine.state.metrics['accuracy']:.4f}")
                
        
        validate_metrics = ['loss', 'accuracy']
        
        for metric in validate_metrics:
            attach_running_average(engine=validate_engine, metric_name=metric)
            
        if verbose >= VERBOSE_epoch_wise:
            @validate_engine.on(Events.EPOCH_COMPLETED) # 학습 데이터의 훈련이 1epoch 끝날때마다
            def print_validate_logs(engine):
                print(f"Epoch: {engine.state.epoch} - loss={engine.state.metrics['loss']:.4e} accuracy={engine.state.metrics['accuracy']:.4f} best_loss={engine.best_loss:.4e}")
        
    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())
            
    @staticmethod
    def save_model(engine, config):
        torch.save(
            {
                'model': engine.best_model,
                'config':config,
            }, config.model_file
        )

                




class Trainer():
    def __init__(self,config):
        self.config = config
        
    def train(self,model,crit,optimizer,train_loader,valid_loader):
        
        train_engine = MyEngine(MyEngine.train, model, crit, optimizer, self.config)
        
        valid_engine = MyEngine(MyEngine.validate, model, crit, optimizer, self.config)
        
        MyEngine.attach(
            train_engine=train_engine,
            validate_engine=valid_engine,
            verbose=self.config.verbose
        )
        
        def run_validate(engine,valid_engine,valid_loader):
            valid_engine.run(valid_loader,max_epochs=1)
        
        
        
        # PyTorch Ignite의 add_event_handler 함수는 특정 이벤트가 발생했을 때 실행할 핸들러 함수(handler)를 등록하는 데 사용됩니다. 이 함수는 Ignite의 핵심 기능 중 하나로, Engine에서 이벤트 기반 작업을 정의할 때 사용됩니다.
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # Events
            run_validate,           # Handler
            valid_engine, valid_loader # Arguments
        )
        
        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # Events
            MyEngine.check_best     # Handler
        )
        
        valid_engine.add_event_handler(
            Events.EPOCH_COMPLETED,  # Events
            MyEngine.save_model,     # Handler
            self.config,             # Arguments
        )
        
        
        train_engine.run(train_loader,max_epochs=self.config.n_epochs)
        
        model.load_state_dict(valid_engine.best_model)
        
        return model
        
