#!/bin/bash

# Save the current date and time to add to the filename
dt=`date '+%Y%m%d_%H%M%S'`

# Create a directory to store the resulting model
mkdir -p ./result_model

# Set parameters
bs=64                # Batch size
epochs=30            # Number of epochs
train_ratio=0.8      # Training data ratio

# Run the Python script
python train.py \
    --model "./result_model/mnist_classification_${dt}_bs${bs}_epoch${epochs}_ratio${train_ratio}.pth" \
    --batch_size $bs \
    --n_epochs $epochs \
    --train_ratio $train_ratio
