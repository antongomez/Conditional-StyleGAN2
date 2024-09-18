# SPDX-FileCopyrightText: 2024 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

import pandas as pd
import argparse
import os
import numpy as np

import matplotlib.pyplot as plt

# This script generates graphics from the data of the training and validation sets
# Given the name of the experiment (for example, MNIST_1), it will return the graphics
# of the loss function, the accuracy of the training and validation sets, and the accuracy
# of each class in the training and validation sets

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Specify the name of the model", required=True)
parser.add_argument("--min-epoch", type=int, help="First epoch to plot loss graphics", default=0)
args = parser.parse_args()

name = args.name
min_epoch = args.min_epoch

# Build filenames
epoch_filename = f"logs/{name}/logs_epoch.csv"
log_filename = f"logs/{name}/logs.csv"
train_filename = f"logs/{name}/train.csv"
val_filename = f"logs/{name}/val.csv"

# Create dir graphics if it doesn't exist
graphics_dir = f"graphics/{name}"
if not os.path.exists(graphics_dir):
    os.makedirs(graphics_dir)

# Graphic epoch loss
df_epoch = pd.read_csv(epoch_filename, delimiter=";", header=0)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
fig.suptitle("Loss function per epoch")

ax1.plot(df_epoch.iloc[min_epoch:, 0], label="Generator Loss")
ax1.plot(df_epoch.iloc[min_epoch:, 1], label="Discriminator Loss")
ax1.legend()
ax1.set_xlabel("Epoch")
ax2.plot(df_epoch.iloc[:, 2], label="Gradient Norm Scaled (by 10)")
ax2.legend()
ax2.set_xlabel("Epoch")
ax3.plot(df_epoch.iloc[:, 3], label="Path Length Regularization")
ax3.legend()
ax3.set_xlabel("Epoch")

plt.savefig(f"{graphics_dir}/epoch_loss.png")

# Graphic epoch loss comparing train and val
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
fig.suptitle("Loss function per epoch on training and validation sets")

ax1.plot(df_epoch.iloc[min_epoch:, 0], label="Generator Loss on Training")
ax1.plot(df_epoch.iloc[min_epoch:, 4], label="Generator Loss on Validation")
ax1.legend()
ax1.set_xlabel("Epoch")
ax2.plot(df_epoch.iloc[min_epoch:, 1], label="Discriminator Loss on Training")
ax2.plot(df_epoch.iloc[min_epoch:, 5], label="Discriminator Loss on Validation")
ax2.legend()
ax2.set_xlabel("Epoch")

plt.savefig(f"{graphics_dir}/epoch_train_val_loss.png")

# Graphic loss
df_loss = pd.read_csv(log_filename, delimiter=";", header=0)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))
fig.suptitle("Loss function in each saved model")

ax1.plot(df_loss.iloc[min_epoch // 2 :, 0], label="Generator Loss")
ax1.plot(df_loss.iloc[min_epoch // 2 :, 1], label="Discriminator Loss")
ax1.legend()
ax1.set_xlabel("Model")
ax2.plot(df_loss.iloc[min_epoch // 2 :, 2], label="Gradient Norm Scaled (by 10)")
ax2.legend()
ax2.set_xlabel("Model")
ax3.plot(df_loss.iloc[min_epoch // 2 :, 3], label="Path Length Regularization")
ax3.legend()
ax3.set_xlabel("Model")

plt.savefig(f"{graphics_dir}/loss.png")

# Graphic train
df_train = pd.read_csv(train_filename, delimiter=";")
class_data = [None] * 10
for i in range(10):
    class_name = f"Class: {i}"
    class_data[i] = df_train.loc[df_train.iloc[:, 1] == class_name]

overall_data = df_train.loc[df_train.iloc[:, 1] == "Overall"]
accuracies_train = {"Overall": np.array([float(ac[:-1]) for ac in overall_data["Accuracy"].values])}
for class_id, class_df in enumerate(class_data):
    accuracies_train[class_id] = np.array([float(ac[:-1]) for ac in class_df["Accuracy"].values])

plt.figure(figsize=(10, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, len(accuracies_train)))
for i, (label, accuracy) in enumerate(accuracies_train.items()):
    plt.plot(accuracy, color=colors[i], label=label)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Global accuracy and accuracy per class on the training set")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.ylim(0, 100)

plt.savefig(f"{graphics_dir}/train_accuracy.png")

# Graphic val
df_val = pd.read_csv(val_filename, delimiter=";")
class_data = [None] * 10
for i in range(10):
    class_name = f"Class: {i}"
    class_data[i] = df_val.loc[df_val.iloc[:, 1] == class_name]

overall_data = df_val.loc[df_val.iloc[:, 1] == "Overall"]
accuracies_val = {"Overall": np.array([float(ac[:-1]) for ac in overall_data["Accuracy"].values])}
for class_id, class_df in enumerate(class_data):
    accuracies_val[class_id] = np.array([float(ac[:-1]) for ac in class_df["Accuracy"].values])

plt.figure(figsize=(10, 6))
colors = plt.cm.rainbow(np.linspace(0, 1, len(accuracies_val)))
for i, (label, accuracy) in enumerate(accuracies_val.items()):
    plt.plot(accuracy, color=colors[i], label=label)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Global accuracy and accuracy per class on the validation set")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.legend()
plt.ylim(0, 100)

plt.savefig(f"{graphics_dir}/val_accuracy.png")

# Generate a graphic with validation and training overall accuracies
plt.figure(figsize=(10, 6))
plt.plot(accuracies_train["Overall"], label="Training")
plt.plot(accuracies_val["Overall"], label="Validation")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Global accuracy on the training and validation sets")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.yticks(np.arange(0, 101, 10))
plt.legend()
plt.ylim(0, 100)

plt.savefig(f"{graphics_dir}/train_val_accuracy.png")
