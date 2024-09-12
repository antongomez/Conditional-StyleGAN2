#!/bin/bash

# SPDX-FileCopyrightText: 2024 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

learning_rates=(0.0002 0.0001 0.00005 0.000025)
# learning_rates_g=(0.0002 0.0001 0.00005 0.000025)
learning_rates_g=(0.000025)

name=$1

# Set parameters
if [ "$name" == "OITAVEN" ]; then
    channels=5
    folder="../data/OITAVEN"
    num_train_steps=250
    evaluate_every=29
    save_every=58
elif [ "$name" == "MNIST" ]; then
    channels=1
    folder="./data"
    num_train_steps=12
    evaluate_every=200
    save_every=250
else
    echo "Invalid dataset name"
    exit 1
fi

echo "Training with learning_rates: ${learning_rates[@]}"
echo "Training with learning_rates_g: ${learning_rates_g[@]}"
echo "Training with network capacities: ${network_capacities[@]}"

k=13
for ((i=0; i<${#learning_rates_g[@]}; i++)); do
    lr_g=${learning_rates_g[i]}
    for ((j=0; j<${#learning_rates[@]}; j++)); do
        lr=${learning_rates[j]}
  
        n=${name}_${k}

        echo "Training with learning rates: $lr and $lr_g (name: $n)"

        python cstylegan2/run.py $folder --channels=$channels --learning_rate=$lr --learning_rate_g=$lr_g --num_train_steps=$num_train_steps --evaluate_every=$evaluate_every --save_every=$save_every --name=$n
        python aux/genDataGraphics.py --name $n
        k=$((k+1))
    done
done