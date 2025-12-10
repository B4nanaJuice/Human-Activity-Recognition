## Import libraries
from dataloader import Dataset

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import json
from sklearn.metrics import confusion_matrix

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Working with {device}")

## Hyperparameters
learning_rate = 1e-3 ## Learning rate for the optimizer
input_size = "52x600"## Size of the input images
epochs = 150         ## Number of epochs for the training
num_runs = 20        ## Number of run, to get an average result at the end 

## Get the data
_train_tensor = torch.load(f'tensors/train_tensors_{input_size}.pt')
_train_labels = torch.load(f'tensors/train_labels_{input_size}.pt')
_test_tensor = torch.load(f'tensors/test_tensors_{input_size}.pt')
_test_labels = torch.load(f'tensors/test_labels_{input_size}.pt')

train_dataset = Dataset(torch.stack(_train_tensor, dim = 0).unsqueeze(1), _train_labels)
test_dataset = Dataset(torch.stack(_test_tensor, dim = 0).unsqueeze(1), _test_labels)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

## Result variables
losses = []          ## Losses during the training part
accuracy = []        ## Accuracy when testing the model
training_time = []   ## Taken time for the training part
conf_matrix = []     ## Average confusion matrix

## Get the model
from models.model_52x600_2conv_1fc import CNN52x600_2CONV1FC

## Make the runs
for run in range(num_runs):
    print(f">>>>> Run [{run+1}/{num_runs}]")

    # Model
    model = CNN52x600_2CONV1FC().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    # Train the model
    losses.append([])
    start_train_time = time.time()
    for epoch in range(epochs):
        print(f"Epoch [{epoch+1}/{epochs}]")

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            scores = model(batch_x)
            loss = criterion(scores, batch_y)
            loss.backward()
            optimizer.step()
        
        losses[-1].append(loss.item())

    end_train_time = time.time()
    _training_time = round(end_train_time - start_train_time, 2)
    training_time.append(_training_time)

    # Test the model
    with torch.no_grad():
        inputs = test_dataset.x.to(device)
        targets = test_dataset.y.to(device)
        
        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim = 1)

        total = targets.size(0)
        correct = (predicted == targets).sum().item()

    _accuracy = round(100 * correct / total, 2)
    accuracy.append(_accuracy)

    conf_matrix.append(confusion_matrix(targets.cpu(), predicted.cpu()))

## Save results in json file
results = {
    "hyperparameters": {
        "learning_rate": learning_rate,
        "input_size": input_size,
        "epochs": epochs,
        "model": type(model).__name__,
        "train_batch_size": 32,
        "test_batch_size": 32,
        "num_runs": num_runs
    },
    "train": {
        "losses": losses,
        "training_time": training_time
    },
    "test": {
        "accuracy": accuracy,
        "confusion_matrix": sum(conf_matrix).tolist()
    }
}

with open(f"results/{type(model).__name__}.json", "w") as f:
    f.write(json.dumps(results, indent = 2))