#!/bin/bash

# SPDX-FileCopyrightText: 2024 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

lr=0.0002
lr_g=0.000025
nc=16

scale_lr=0.125
scale_lr_g=0.75

dataset=$1
k=$2
name="$1_$2"

# Set parameters
if [ "$dataset" == "OITAVEN" ]; then
    channels=5
    folder="./data/OITAVEN"
    num_train_steps=200
    evaluate_every=50
    save_every=100
elif [ "$dataset" == "XESTA" ]; then
    channels=5
    folder="./data/XESTA"
    num_train_steps=100
    evaluate_every=143
    save_every=143
elif [ "$dataset" == "EIRAS" ]; then
    channels=5
    folder="./data/EIRAS"
    num_train_steps=200
    evaluate_every=36
    save_every=72
elif [ "$dataset" == "ERMIDAS" ]; then
    channels=5
    folder="./data/ERMIDAS"
    num_train_steps=200
    evaluate_every=54
    save_every=108
elif [ "$dataset" == "FERREIRAS" ]; then
    channels=5
    folder="./data/FERREIRAS"
    num_train_steps=100
    evaluate_every=127
    save_every=127
elif [ "$dataset" == "MESTAS" ]; then
    channels=5
    folder="./data/MESTAS"
    num_train_steps=200
    evaluate_every=91
    save_every=182
elif [ "$dataset" == "MERA" ]; then
    channels=5
    folder="./data/MERA"
    num_train_steps=100
    evaluate_every=140
    save_every=140
elif [ "$dataset" == "ULLA" ]; then
    channels=5
    folder="./data/ULLA"
    num_train_steps=800
    evaluate_every=12
    save_every=24
elif [ "$dataset" == "MNIST" ]; then
    channels=1
    folder="./data"
    num_train_steps=12
    evaluate_every=200
    save_every=250
else
    echo "Invalid dataset name"
    exit 1
fi

lr_final=$(echo "scale=9; $lr * ${scale_lr}" | bc)
lr_g_final=$(echo "scale=9; $lr_g * ${scale_lr_g}" | bc)

echo "Training with learning_rates: $lr and $lr_g. Network capacity: $nc (name $name)"
echo "Final learning rates: $lr_final and $lr_g_final"

python cstylegan2/run.py $folder --channels=$channels --learning_rate_final=$lr_final --learning_rate_g_final=$lr_g_final --learning_rate=$lr --learning_rate_g=$lr_g --network-capacity=$nc --num_train_steps=$num_train_steps --evaluate_every=$evaluate_every --save_every=$save_every --name=$name
python aux/genDataGraphics.py --name $name --min-epoch=4
