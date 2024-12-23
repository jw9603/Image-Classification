#!/bin/bash

# Save the current date and time to add to the filename
dt=`date '+%Y%m%d_%H%M%S'`

# Create a directory to store the resulting model
mkdir -p ./result_model

# Set parameters
bs=256               # Batch size
epochs=3             # Number of epochs
train_ratio=0.8      # Training data ratio
model_name='rnn'     # fc, cnn, and rnn

# RNN-specific parameters
hidden_size=128
n_layers=4
dropout_p=0.2

# Construct the Python command
python_command="python train.py \
    --model_file \"./result_model/mnist_classification_${model_name}_${dt}_bs${bs}_epoch${epochs}_ratio${train_ratio}.pth\" \
    --model $model_name \
    --batch_size $bs \
    --n_epochs $epochs \
    --train_ratio $train_ratio"

# Add RNN-specific arguments if model_name is 'rnn'
if [ "$model_name" == "rnn" ]; then
    python_command+=" \
    --hidden_size $hidden_size \
    --n_layers $n_layers \
    --dropout_p $dropout_p"
fi

# Execute the Python command
eval $python_command
