"""
model_full_run will be the same as model_run but for each run, the datasets will be 
regenerated to ensure each run is completely independant. 
"""

## Import libs
from dataloader import Dataset
from model_utils import train_model

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import json

from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working with {device}")

## Hyperparameters
input_size = "64x64"    ## Size of the input images
epochs = 150             ## Number of epochs for the training
runs = 20                ## Number of run, to get an average result at the end 

## Get tensors
tensors = torch.stack(torch.load(f'tensors/full/tensors_{input_size}.pt'))
labels = torch.load('tensors/full/labels.pt')

## Result variables
losses = []              ## Losses during the training part
accuracy = []            ## Accuracy when testing the model
tt = []                  ## Taken time for the training part
cm = []                  ## Average confusion matrix

## Import the model
from models.model_64x64_2conv_1fc import CNN64x64_2CONV1FC

## Make the runs
for run in range(runs):
    print(f'>>>>> Run [{run+1}/{runs}]')

    # Split the data and create datasets
    test_rows = torch.randint(len(labels)-1, (int(.2 * len(labels)),))
    train_rows = torch.from_numpy(np.array([_ for _ in range(len(labels)) if _ not in test_rows]))
    
    test_tensors = tensors[test_rows]
    test_labels = labels[test_rows]

    train_tensors = tensors[train_rows]
    train_labels = labels[train_rows]

    train_dataset = Dataset(train_tensors.unsqueeze(1), train_labels)
    test_dataset = Dataset(test_tensors.unsqueeze(1), test_labels)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)

    # Import model
    model = CNN64x64_2CONV1FC().to(device)

    # Train the model
    model, loss, ttime = train_model(model = model, loader = train_loader, epochs = epochs)
    losses.append(loss)
    tt.append(ttime)

    # Evaluate the model
    with torch.no_grad():
        inputs = test_dataset.x.to(device)
        targets = test_dataset.y.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim = 1)

        total = targets.size(0)
        correct = (predicted == targets).sum().item()
    
    accuracy.append(round(100 * correct / total, 2))
    cm.append(confusion_matrix(targets.cpu(), predicted.cpu()))

## Save results in json file
results = {
    "hyperparameters": {
        "learning_rate": 1e-3,
        "input_size": input_size,
        "epochs": epochs,
        "model": type(model).__name__,
        "train_batch_size": 32,
        "test_batch_size": 32,
        "num_runs": runs
    },
    "train": {
        "losses": losses,
        "training_time": tt
    },
    "test": {
        "accuracy": accuracy,
        "confusion_matrix": sum(cm).tolist()
    }
}

with open(f"results/{type(model).__name__}.json", "w") as f:
    f.write(json.dumps(results, indent = 2))