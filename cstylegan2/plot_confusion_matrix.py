# SPDX-FileCopyrightText: 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os

from trainer import Trainer
from dataset import DatasetManager

from torch.utils import data
import numpy as np
import argparse

import matplotlib.pyplot as plt


def show_confusion_matrix(confusion_matrix, save_folder, model_number, validation=False):
    confusion_matrix = np.array(confusion_matrix)
    print(confusion_matrix)

    # reset plot
    plt.figure()

    # Plot confusion matrix as heatmap
    plt.imshow(confusion_matrix, cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    plt.colorbar()

    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(
                j, i, confusion_matrix[i, j], horizontalalignment="center", verticalalignment="center", color="black"
            )

    image_name = f"cm_val" if validation else f"cm"
    if model_number != -1:
        plt.savefig(f"{save_folder}/{image_name}_{model_number}.png")
    else:
        plt.savefig(f"{save_folder}/{image_name}.png")

    # close figure
    plt.close()


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name argument", required=True)
parser.add_argument("--model", type=int, help="Model number", default=-1)

args = parser.parse_args()
name = args.name
model_number = args.model

root = "./models"

dataset_name = name.split("_")[0]
if dataset_name == "MNIST" or dataset_name == "MNIST2":
    folder = f"./data"
else:
    folder = f"./data/{dataset_name}"

save_folder = f"./test/{name}/D"

os.makedirs(save_folder, exist_ok=True)

with open(os.path.join(root, name, "config.json"), "r") as file:
    config = json.load(file)

batch_size = config["batch_size"]
channels = config["channels"]
image_size = config["image_size"]

dataset_manager = DatasetManager(folder, train=False, hyperdataset=(True if channels > 4 else False))
test_dataset = dataset_manager.get_test_set()
print("Dataset length:", len(test_dataset))
test_loader = data.DataLoader(
    test_dataset, num_workers=0, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=False
)

if channels > 4:
    val_dataset = dataset_manager.get_validation_set()
    print("Validation dataset length:", len(val_dataset))
    val_loader = data.DataLoader(
        val_dataset, num_workers=0, batch_size=batch_size, drop_last=True, shuffle=True, pin_memory=False
    )
else:
    val_loader = None

model = Trainer(**config, label_dim=test_dataset.label_dim)
model.load(model_number, root=root)

# Calculate accuracy
correct_per_class, total_per_class, confusion_matrix = model.calculate_accuracy(
    test_loader, show_progress=True, confusion_matrix=True
)

# Imprimimos a accuracy por clase
print(
    f"Overall accuracy: {sum(correct_per_class)}/{sum(total_per_class)} ({sum(correct_per_class)/sum(total_per_class):.2%})"
)
for i, (correct, total) in enumerate(zip(correct_per_class, total_per_class)):
    print(f"Class {i}: {correct}/{total} ({correct/total:.2%})")
# Imprimimos a matriz de confusion en cor
show_confusion_matrix(confusion_matrix, save_folder, model_number)

# Calculate accuracy on validation set
if val_loader:
    correct_per_class, total_per_class, confusion_matrix = model.calculate_accuracy(
        val_loader, show_progress=True, confusion_matrix=True
    )

    # Imprimimos a accuracy por clase
    print(
        f"Overall accuracy: {sum(correct_per_class)}/{sum(total_per_class)} ({sum(correct_per_class)/sum(total_per_class):.2%})"
    )
    for i, (correct, total) in enumerate(zip(correct_per_class, total_per_class)):
        print(f"Class {i}: {correct}/{total} ({correct/total:.2%})")
    # Imprimimos a matriz de confusion en cor
    show_confusion_matrix(confusion_matrix, save_folder, model_number, validation=True)
