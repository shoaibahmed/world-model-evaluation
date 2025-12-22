#!/bin/bash

epochs=1
python train.py --model_name "next-tok-ep"${epochs} --max_epochs ${epochs} --use_wandb True
python train.py --model_name "next-lat-pad-fixed-ep"${epochs} --max_epochs ${epochs} --next_lat_pred True --use_wandb True

# Perform detour analysis
python detour_analysis.py --model-name "next-tok-ep"${epochs}
python detour_analysis.py --model-name "next-lat-pad-fixed-ep"${epochs} --next-lat-pred
