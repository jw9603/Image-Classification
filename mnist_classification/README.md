# MNIST Classification

This directory contains code for training and testing the MNIST dataset using only fully connected layers.

The code is designed to work with any dataset as long as you adjust the dataloader section and the input and output sizes.


## Usage

### 0. File Structure

The file structure will look like:

```plain
.
├── README.md
├── mnist_classification/
    ├── resut_model/                 (Model_File)
    ├── model.py/                    (Fully Connected Layer)
    ├── trainer.py/
    ├── train.py/
    ├── utils.py/
    └── run.sh/
```



### 1. Train
```
bash ./run.sh
```

### 2. Evaluate trained model
```
See test.ipynb
```
