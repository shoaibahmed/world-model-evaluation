#!/bin/bash

epochs=1
python train.py --model_name "next-tok-ep"${epochs} --max_epochs ${epochs} --use_wandb True
python train.py --model_name "next-lat-pad-fixed-ep"${epochs} --max_epochs ${epochs} --next_lat_pred True --use_wandb True
