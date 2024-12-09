# MNIST Classification

This directory contains code for training and testing the MNIST dataset using fully connected layers only.

The code is designed to be flexible and can work with any dataset, provided that the data loader and the model's input and output sizes are appropriately adjusted.

I have structured a training boilerplate using [PyTorch Ignite](https://pytorch.org/ignite/index.html), enabling streamlined training and testing.

By modifying the data loader and the model's input and output, this code can be adapted to train on various datasets.


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

### 1. Models

- MLP (Fully Connected Layers)
- CNN
- RNN


### 2. Train
```
bash ./run_train.sh
```

### 3. Evaluate trained model
```
python ./test.py
```
