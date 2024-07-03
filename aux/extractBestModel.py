import pandas as pd
import argparse
import numpy as np

# This script extracts the best model from the validation data
# Given the name of the experiment (for example, MNIST_1), it will return the best model
# based on the average accuracy of the validation data

parser = argparse.ArgumentParser()
parser.add_argument("--name", help="Specify the name of the model", required=True)
args = parser.parse_args()

name = args.name

# Build filenames
val_filename = f"logs/{name}/val.csv"

# Load data
df_val = pd.read_csv(val_filename, delimiter=";")
number_of_models = int(df_val.iloc[-1, 0]) + 1

# Calculate overall accuracies
overall_data = df_val.loc[df_val.iloc[:, 1] == "Overall"]
overall_accuracies = np.array([float(ac[:-1]) for ac in overall_data["Accuracy"].values])

# Calculate average accuracies
average_accuracies = np.zeros((number_of_models,))
for i in range(number_of_models):
    s = df_val.loc[df_val.iloc[:, 0] == i]
    # mean all of the values of accuracy
    accuracies = np.array([float(ac[:-1]) for ac in s["Accuracy"].values])
    average_accuracies[i] = accuracies.mean()

# Calculate max overall accuracy and max average accuracy
max_average_accuracy = average_accuracies.max()

# Get the index of max overall accuracy and max average accuracy
max_overall_accuracy_index = overall_accuracies.argmax() + 1
max_average_accuracy_index = average_accuracies.argmax() + 1

overall_accuracy_aa = overall_accuracies[max_average_accuracy_index]

print("OA: ", round(overall_accuracy_aa, 2), sep="")
print("AA: ", round(max_average_accuracy, 2), sep="")
print(
    "Model: ",
    max_average_accuracy_index * 2,
    "/",
    number_of_models * 2,
    "  Real model: ",
    max_average_accuracy_index - 1,
    sep="",
)
