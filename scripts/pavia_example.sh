#!/bin/bash

# SPDX-FileCopyrightText: 2024 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

num_train_steps=12
evaluate_every=44 # evaluate image generation each epoch
save_every=44 # save a model each epoch
name=PAVIA

echo "Training steps $num_train_steps (name: $name)"

python cstylegan2/run.py data/PAVIA --channels=5 --num_train_steps=$num_train_steps --evaluate_every=$evaluate_every --save_every=$save_every --name=$name
