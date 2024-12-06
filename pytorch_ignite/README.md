# MNIST Classification

This directory contains code for training and testing the MNIST dataset using only fully connected layers.

The code is designed to work with any dataset as long as you adjust the dataloader section and the input and output sizes.

I structured a training boilerplate using PyTorch Ignite.


## Usage

### 0. File Structure

The file structure will look like:

```plain
.
├── README.md
├── pytorch_ignite/
    ├── data/
    ├── result_mode/                
    ├── data_loader.py/
    ├── model.py/
    ├── run_train.sh/
    ├── test.py/
    ├── train.py/
    ├── trainer.py/
```



### 1. Train
```
bash ./run_train.sh
```

### 2. Evaluate trained model
```
python ./test.py
```
