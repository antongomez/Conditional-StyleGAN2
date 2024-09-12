#!/bin/bash

# SPDX-FileCopyrightText: 2024 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

learning_rates=(0.0002 0.0002 0.0002)
learning_rates_g=(0.00005 0.000025 0.000025)
network_capacities=(16 16 32)

scales_lr=(0.25 0.125 0.125)
scales_lr_g=(1 0.75 0.5)

name=$1

# Set parameters
if [ "$name" == "OITAVEN" ]; then
    channels=5
    folder="./data/OITAVEN"
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

k=25
for ((i=0; i<${#learning_rates_g[@]}; i++)); do
    lr_g=${learning_rates_g[i]}
    lr=${learning_rates[i]}
    nc=${network_capacities[i]}

    for ((j=0; j<${#scales_lr[@]}; j++)); do
        lr_final=$(echo "scale=9; $lr * ${scales_lr[j]}" | bc)
        lr_g_final=$(echo "scale=9; $lr_g * ${scales_lr_g[j]}" | bc)

        n=${name}_${k}

        echo "Training with learning rates: $lr and $lr_g. Network capacity: $nc (name: $n)"
        echo "Final learning rates: $lr_final and $lr_g_final"

        python cstylegan2/run.py $folder --channels=$channels --learning_rate_final=$lr_final --learning_rate_g_final=$lr_g_final --learning_rate=$lr --learning_rate_g=$lr_g --num_train_steps=$num_train_steps --evaluate_every=$evaluate_every --save_every=$save_every --name=$n
        python aux/genDataGraphics.py --name $n --min-epoch=4
        k=$((k+1))
    done
done