#!/bin/bash

# Save the current date and time to add to the filename
dt=`date '+%Y%m%d_%H%M%S'`

# Create a directory to store the resulting model
mkdir -p ./result_model

# Set parameters
bs=256               # Batch size
epochs=5          # Number of epochs
train_ratio=0.8      # Training data ratio
model_name='cnn'     # fc or cnn

# Run the Python script
python train.py \
    --model_file "./result_model/mnist_classification_${model_name}_${dt}_bs${bs}_epoch${epochs}_ratio${train_ratio}.pth" \
    --model $model_name \
    --batch_size $bs \
    --n_epochs $epochs \
    --train_ratio $train_ratio
