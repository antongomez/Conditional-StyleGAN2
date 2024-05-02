import os
import json

from trainer import Trainer

import numpy as np
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='Name')
args = parser.parse_args()

name = args.name

root = './models'
folder = './data'

save_generated = f'./test/{name}/G/generated'
save_average_generated = f'./test/{name}/G/average_generated'

os.makedirs(save_generated, exist_ok=True)
os.makedirs(save_average_generated, exist_ok=True)

with open(os.path.join(root, name, 'config.json'), 'r') as file:
    config = json.load(file)
model = Trainer(**config)
model.load(-1, root=root)  # the first argument is the index of the checkpoint, -1 means the last checkpoint

mode = 'L' if config['channels'] == 1 else 'RGB'

labels_to_evaluate = np.array([np.eye(10)[i % 10] for i in range(50)])


model.set_evaluation_parameters(labels_to_evaluate=labels_to_evaluate, reset=True, total=len(labels_to_evaluate))  # you can set the latents, the noise or the labels
generated_images, average_generated_images = model.evaluate()

def save_image(tensor_image, path, mode):
    tensor_image = tensor_image.cpu()
    if(mode == "L"):
        tensor_image = tensor_image.squeeze() # Pillow expects HxW with mode L
    numpy_image = tensor_image.detach().numpy()
    numpy_image = (numpy_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(numpy_image, mode=mode)
    pil_image.save(path)

# Save the images
os.makedirs('./results', exist_ok=True)
for i, tensor_image in enumerate(generated_images):
    save_image(tensor_image, f'{save_generated}/class_{i % 10}_{i}.png', mode)

for i, tensor_image in enumerate(average_generated_images):
    save_image(tensor_image, f'{save_average_generated}/image_{i % 10}_{i}.png', mode)