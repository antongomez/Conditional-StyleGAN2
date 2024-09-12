# SPDX-FileCopyrightText: 2024 2024, Antón Gómez López
#
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import os

from trainer import Trainer
from dataset import DatasetManager

from torch.utils import data
import argparse

# This script calculates pixel accuracy remmoving the
# centrer of the segments used for training and validation

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, help="Folder argument", required=True)
parser.add_argument("--name", type=str, help="Name argument", required=True)
parser.add_argument("--model", type=int, help="Model number", default=-1)

args = parser.parse_args()
folder = args.folder
name = args.name
model_number = args.model

root = "./models"
save_folder = f"./test/{name}/D"

os.makedirs(save_folder, exist_ok=True)

with open(os.path.join(root, name, "config.json"), "r") as file:
    config = json.load(file)

batch_size = config["batch_size"]
channels = config["channels"]
image_size = config["image_size"]

dataset_manager = DatasetManager(
    folder,
    train=False,
    hyperdataset=(True if channels > 4 else False),
    batch_size=batch_size,
)
test_dataset = dataset_manager.get_test_set()
print("Dataset length:", len(test_dataset))
test_loader = data.DataLoader(
    test_dataset, num_workers=0, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=False
)

if channels > 4:
    val_dataset = dataset_manager.get_validation_set()
    print("Validation dataset length:", len(val_dataset))
    val_loader = data.DataLoader(
        val_dataset, num_workers=0, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=False
    )
else:
    val_loader = None

model = Trainer(**config, label_dim=test_dataset.label_dim)
model.load(model_number, root=root)

OA, AA, class_accuracies = model.calculate_pixel_accuracy(
    dataset_manager, test_dataset, test_loader, show_progress=True
)

print(f"Overall accuracy: {OA:.2%}")
print(f"Average accuracy: {AA:.2%}")
for i, acc in enumerate(class_accuracies[1:]):
    print(f"Class {i}: {acc:.2%}")
