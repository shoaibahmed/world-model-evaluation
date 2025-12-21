#!/bin/bash

python train.py --model_name "next-tok" --use_wandb True
python train.py --model_name "next-lat" --next_lat_pred --use_wandb True
