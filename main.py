from dataloader import Dataset, csv_to_tensor
from model import CNN
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Import data and create data loader
data_path = 'data'
_result: list[tuple[str, torch.Tensor]] = []
for _sub_directory in os.scandir(data_path):

    # Make sure the path points to a folder and not a file
    if not _sub_directory.is_dir():
        pass

    _label: str = _sub_directory.name
    for _file in os.scandir(f'{data_path}\\{_sub_directory.name}'):
        _result.append((_label, csv_to_tensor(f'{data_path}\\{_sub_directory.name}\\{_file.name}')))

# Compute the number of rows that will be used to train the model
test_percentage = .1
_test_rows = int(len(_result) * test_percentage)

# Shuffle the data and take the rows
np.random.shuffle(_result)
_train, _test = _result[_test_rows:], _result[:_test_rows]

label_to_id = {
    'bend': 0,
    'fall': 1,
    'lie down': 2,
    'run': 3,
    'sitdown': 4,
    'standup': 5,
    'walk': 6,
}

_train_tensor, _train_labels = [_[1] for _ in _train], torch.from_numpy(np.array([label_to_id[_[0]] for _ in _train]))
_test_tensor, _test_labels = [_[1] for _ in _test], torch.from_numpy(np.array([label_to_id[_[0]] for _ in _test]))

train_dataset = Dataset(torch.stack(_train_tensor, dim = 0).unsqueeze(1), _train_labels)
test_dataset = Dataset(torch.stack(_test_tensor, dim = 0).unsqueeze(1), _test_labels)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

print(f'Training on : {device}')

losses = []
num_epochs = 75
for epoch in range(num_epochs):
    if (epoch+1) % 5 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}] - ', end = '')

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        scores = model(batch_x)
        loss = criterion(scores, batch_y)
        loss.backward()
        optimizer.step()
    
    losses.append(loss.item())
    if (epoch+1) % 5 == 0:  
        print(f'loss : {loss.item():.4f}')

# Test the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs, dim = 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100*correct / total
print(f'Accuracy : {accuracy:.2f}')
