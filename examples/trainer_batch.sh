#!/bin/bash

hidden_dims_list=("[16]" "[32]" "[16,32]")
learning_rates=(0.001 0.0001)
epoch_max_list=(20 50)

mkdir -p logs  # Create a folder for logs

for hidden_dims in "${hidden_dims_list[@]}"; do
    for lr in "${learning_rates[@]}"; do
        for epoch_max in "${epoch_max_list[@]}"; do
            logfile="logs/run_${hidden_dims}_${lr}_${epoch_max}.log"
            python3 MLPexample.py --hidden_dims $hidden_dims --lr $lr --epoch_max $epoch_max > "$logfile" 2>&1
        done
    done
done
