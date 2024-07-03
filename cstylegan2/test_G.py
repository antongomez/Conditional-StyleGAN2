import os
import json
import argparse

from trainer import Trainer

import numpy as np
from torchvision.utils import save_image
from torch import tensor, index_select

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, help="Name")
parser.add_argument("--model", type=int, help="Model number", default=-1)
args = parser.parse_args()

name = args.name
model_number = args.model

root = "./models"
folder = "./data"

save_generated = f"./test/{name}/G/generated"
save_average_generated = f"./test/{name}/G/average_generated"

os.makedirs(save_generated, exist_ok=True)
os.makedirs(save_average_generated, exist_ok=True)

with open(os.path.join(root, name, "config.json"), "r") as file:
    config = json.load(file)
model = Trainer(**config)
model.load(model_number, root=root)  # the first argument is the index of the checkpoint, -1 means the last checkpoint


hyperdataset = True if config["channels"] > 4 else False

# 5 images of each class
labels_to_evaluate = np.array([np.eye(10)[i % 10] for i in range(50)])


model.set_evaluation_parameters(
    labels_to_evaluate=labels_to_evaluate, reset=True, total=len(labels_to_evaluate)
)  # you can set the latents, the noise or the labels
generated_images, average_generated_images = model.evaluate()


def tensor_index_select(tensor, dim=0, index=tensor([2, 1, 0])):
    return index_select(tensor, dim=dim, index=index) if hyperdataset else tensor


# Save the images
os.makedirs("./results", exist_ok=True)
for i, tensor_image in enumerate(generated_images):
    save_image(tensor_index_select(tensor_image.cpu()), f"{save_generated}/class_{i % 10}_{i}.png")

for i, tensor_image in enumerate(average_generated_images):
    save_image(tensor_index_select(tensor_image.cpu()), f"{save_average_generated}/image_{i % 10}_{i}.png")
