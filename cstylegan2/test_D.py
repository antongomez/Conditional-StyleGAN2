import json
import os

from trainer import Trainer
from dataset import DatasetManager

from torch.utils import data
from torch import abs as torch_abs
from torch import linspace as torch_linspace
import numpy as np
import argparse

import matplotlib.pyplot as plt

def show_confusion_matrix(confusion_matrix, save_folder, model_number, validation=False):
    confusion_matrix = np.array(confusion_matrix)
    print(confusion_matrix)

    # reset plot
    plt.figure()

    # Plot confusion matrix as heatmap
    plt.imshow(confusion_matrix, cmap='Blues')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')

    plt.colorbar()

    # Add text annotations
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, confusion_matrix[i, j],
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='black')
            
    image_name = f'cm_val' if validation else f'cm'
    if model_number != -1:
        plt.savefig(f'{save_folder}/{image_name}_{model_number}.png')
    else:
        plt.savefig(f'{save_folder}/{image_name}.png')

    # close figure
    plt.close()

# Calculate the line equation given the weights and bias
def calculate_line(x, weight, bias):
    return (-weight[0]/weight[1]) * x - bias/weight[1]

# Plot hyperplanes given the weights and biases. Also plot the first k points of each class
def plot_points_class(save_folder, model_number, weights, biases, label, point_list, hyperplanes=True):
    if hyperplanes:
        print(f'Plotting hyperplanes for class {label}')
        print(weights)
        print(biases)

    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    x = torch_linspace(-2.5, 2.5, 20)

    # reset plot
    plt.figure()

    xmin = -2.5
    xmax = 2.5
    ymin = -2.5
    ymax = 2.5

    for i in range(10):
        if hyperplanes:
            y = calculate_line(x, weights[i, :], biases[i])
            plt.plot(x, y, color=colors[i])
        plt.scatter(point_list[i][:, 0], point_list[i][:, 1], color=colors[i], label=f'Class {i}', s=10)

        if min(point_list[i][:, 0]) < xmin:
            xmin = min(point_list[i][:, 0])
        if max(point_list[i][:, 0]) > xmax:
            xmax = max(point_list[i][:, 0])
        if min(point_list[i][:, 1]) < ymin:
            ymin = min(point_list[i][:, 1])
        if max(point_list[i][:, 1]) > ymax:
            ymax = max(point_list[i][:, 1])

    plt.xlim(xmin, xmax)
    if hyperplanes:
        plt.ylim(-700, 600)
    else:
        plt.ylim(ymin, ymax)
    plt.legend()

    image_name = f'hyperplanes_{label}.png' if hyperplanes else f'points_{label}.png'
    image_name = f'{model_number}_{image_name}' if model_number != -1 else image_name
    plt.savefig(f'{save_folder}/{image_name}')

    # close figure
    plt.close()

# Get the first k points of a given class
def get_first_k_points_label(intermediate_outputs, labels, k, label):
    indices_label = np.where(labels == label)[0]
    return intermediate_outputs[indices_label[:k], :]

# Plot the hyperplanes for each class
def plot_hyperplanes(save_folder, model_number, weights, biases, indices, intermediate_outputs, intermediate_labels):
        k = 5
        for label in range(10):
            point_list = [get_first_k_points_label(intermediate_outputs[:, indices[label, ]], intermediate_labels, k, i) for i in range(10)]
            plot_points_class(save_folder, model_number, weights[:, indices[label, ]], biases, label, point_list, hyperplanes=True)
            plot_points_class(save_folder, model_number, weights[:, indices[label, ]], biases, label, point_list, hyperplanes=False)

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='Name argument', required=True)
parser.add_argument('--model', type=int, help='Model number', default=-1)

args = parser.parse_args()
name = args.name
model_number = args.model

root = './models'

dataset_name = name.split("_")[0]
if dataset_name == 'MNIST' or dataset_name=='MNIST2':    
    folder = f'./data'
else:
    folder = f'./data/{dataset_name}'

save_folder = f'./test/{name}/D'

os.makedirs(save_folder, exist_ok=True)

with open(os.path.join(root, name, 'config.json'), 'r') as file:
    config = json.load(file)

batch_size = config['batch_size']
channels = config['channels']
image_size = config['image_size']

dataset_manager = DatasetManager(folder, train=False, hyperdataset=(True if channels > 4 else False))
test_dataset = dataset_manager.get_test_set()
print("Dataset length:", len(test_dataset))
test_loader = data.DataLoader(test_dataset, num_workers=0, batch_size=batch_size,
                                    drop_last=True, shuffle=True, pin_memory=False)

if channels > 4:
    val_dataset = dataset_manager.get_validation_set()
    print("Validation dataset length:", len(val_dataset))
    val_loader = data.DataLoader(val_dataset, num_workers=0, batch_size=batch_size,
                                    drop_last=True, shuffle=True, pin_memory=False)
else:
    val_loader = None

model = Trainer(**config, label_dim=test_dataset.label_dim)
model.load(model_number, root=root)

# Calculate accuracy
correct_per_class, total_per_class, confusion_matrix = model.calculate_accuracy(test_loader, show_progress=True, confusion_matrix=True)

# Imprimimos a accuracy por clase
print(f'Overall accuracy: {sum(correct_per_class)}/{sum(total_per_class)} ({sum(correct_per_class)/sum(total_per_class):.2%})')
for i, (correct, total) in enumerate(zip(correct_per_class, total_per_class)):
    print(f'Class {i}: {correct}/{total} ({correct/total:.2%})')
# Imprimimos a matriz de confusion en cor
show_confusion_matrix(confusion_matrix, save_folder, model_number)

# Calculate accuracy on validation set
if val_loader:
    correct_per_class, total_per_class, confusion_matrix = model.calculate_accuracy(val_loader, show_progress=True, confusion_matrix=True)

    # Imprimimos a accuracy por clase
    print(f'Overall accuracy: {sum(correct_per_class)}/{sum(total_per_class)} ({sum(correct_per_class)/sum(total_per_class):.2%})')
    for i, (correct, total) in enumerate(zip(correct_per_class, total_per_class)):
        print(f'Class {i}: {correct}/{total} ({correct/total:.2%})')
    # Imprimimos a matriz de confusion en cor
    show_confusion_matrix(confusion_matrix, save_folder, model_number, validation=True)


# Calculate the intermediate representation of the dataset
# intermediate_outputs, labels, weights, biases = model.get_intermediate_representation(test_loader, show_progress=True)
# # Pasamos as etiquetas de one hot a enteiros
# labels = labels.argmax(dim=1)

# # Find the top 2 weights in absolute value for each SVM classifier
# values, indices = torch_abs(weights).topk(2, dim=1, largest=False)
# print(values)
# print(indices)
# print(biases)

# plot_hyperplanes(save_folder, model_number, weights.cpu(), biases.cpu(), indices.cpu(), intermediate_outputs.cpu(), labels.cpu())
