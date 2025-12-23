#!/bin/bash

epochs=1
data="shortest-paths"

base_args="--data ${data}"
next_tok_args="${base_args} --model-name ${data}-next-tok-ep${epochs}"
next_lat_args="${base_args} --model-name ${data}-next-lat-ep${epochs} --next-lat-pred"

# Train models
python train.py ${next_tok_args} --max-epochs ${epochs} --use-wandb
python train.py ${next_lat_args} --max-epochs ${epochs} --use-wandb

# Perform next-token test
python nex_token_test.py ${next_tok_args}
python nex_token_test.py ${next_lat_args}

# Perform current state probe test
python probe_test.py ${next_tok_args}
python probe_test.py ${next_lat_args}

# Perform compression test
python compression_test.py ${next_tok_args}
python compression_test.py ${next_lat_args}

# Perform distinction test
python distinction_test.py ${next_tok_args}
python distinction_test.py ${next_lat_args}

# Perform detour analysis
python detour_analysis.py ${next_tok_args}
python detour_analysis.py ${next_lat_args}
