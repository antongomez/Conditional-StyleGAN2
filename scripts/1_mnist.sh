#!/bin/bash

# SPDX-FileCopyrightText: 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

# learning_rates=(0.0002 0.0001)
# learning_rates_g=(0.0002 0.0001)
learning_rates=(0.0002)
learning_rates_g=(0.0002)

name=$1

# Set parameters
if [ "$name" == "MNIST" ]; then
    channels=1
    folder="./data"
    num_train_steps=12
    evaluate_every=200
    save_every=250
    val_size=1000
else
    echo "Invalid dataset name"
    exit 1
fi

echo "Training with learning_rates: ${learning_rates[@]}"
echo "Training with learning_rates_g: ${learning_rates_g[@]}"
echo "Training with network capacities: ${network_capacities[@]}"

k=1
for ((i=0; i<${#learning_rates_g[@]}; i++)); do
    lr_g=${learning_rates_g[i]}
    for ((j=0; j<${#learning_rates[@]}; j++)); do
        lr=${learning_rates[j]}
  
        n=${name}_${k}

        echo "Training with learning rates: $lr and $lr_g (name: $n)"

        python cstylegan2/run.py $folder --channels=$channels --learning_rate=$lr --learning_rate_g=$lr_g --num_train_steps=$num_train_steps --evaluate_every=$evaluate_every --save_every=$save_every --val_size=$val_size --name=$n
        python aux/gen_data_graphics.py --name $n
        k=$((k+1))
    done
done