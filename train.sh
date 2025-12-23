#!/bin/bash

epochs=1
data="shortest-paths"

base_args="--data ${data}"
next_tok_args="${base_args} --model_name ${data}-next-tok-ep${epochs}"
next_lat_args="${base_args} --model_name ${data}-next-lat-ep${epochs}"

# Train models
python train.py ${next_tok_args} --max_epochs ${epochs} --use_wandb True
python train.py ${next_lat_args} --next_lat_pred True --max_epochs ${epochs} --use_wandb True

# Perform detour analysis
python detour_analysis.py ${next_tok_args}
python detour_analysis.py ${next_lat_args} --next-lat-pred

# Perform compression test
python compression_test.py ${next_tok_args}
python compression_test.py ${next_lat_args} --next-lat-pred
