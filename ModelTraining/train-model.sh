#!/bin/bash -v
# The name of the model to train, e.g. "model-1" is $1
# A directory with this name will be created in the current directory
# and the training script, output and model will be stored there.

mkdir ./$1
if [ $? -ne 0 ]; then
    echo "Directory $1 already exists"
    exit 1
fi
cp Analysis/train_model.py ./$1
if [ $? -ne 0 ]; then
    echo "Could not copy train_model.py to $1"
    exit 1
fi
nohup ./Analysis/train_model.py ./$1 > ./$1/train-output.log 2>&1 &

