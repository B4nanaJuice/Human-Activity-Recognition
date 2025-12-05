## Import libraries
from dataloader import Dataset
from model_square_images import CNNSquareImages
from model_unmodified_images import CNNUntouchedImages

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
from sklearn import metrics

## Hyperparameters
batch_size_train = 32
batch_size_test = 42
learning_rate = 1e-3
epochs = 500

## Define used device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using {device}')

## Import images (inputs for the CNNs)
_train_tensor_52x600 = torch.load('tensors/train_tensors_52x600.pt')
_train_labels_52x600 = torch.load('tensors/train_labels_52x600.pt')
_test_tensor_52x600 = torch.load('tensors/test_tensors_52x600.pt')
_test_labels_52x600 = torch.load('tensors/test_labels_52x600.pt')

_train_tensor_64x64 = torch.load('tensors/train_tensors_64x64.pt')
_train_labels_64x64 = torch.load('tensors/train_labels_64x64.pt')
_test_tensor_64x64 = torch.load('tensors/test_tensors_64x64.pt')
_test_labels_64x64 = torch.load('tensors/test_labels_64x64.pt')

# Images 52 x 600
train_dataset_52x600 = Dataset(torch.stack(_train_tensor_52x600, dim = 0).unsqueeze(1), _train_labels_52x600)
test_dataset_52x600 = Dataset(torch.stack(_test_tensor_52x600, dim = 0).unsqueeze(1), _test_labels_52x600)

train_loader_52x600 = DataLoader(train_dataset_52x600, batch_size = batch_size_train, shuffle = True)
test_loader_52x600 = DataLoader(test_dataset_52x600, batch_size = batch_size_test, shuffle = True)

# Images 64 x 64
train_dataset_64x64 = Dataset(torch.stack(_train_tensor_64x64, dim = 0).unsqueeze(1), _train_labels_64x64)
test_dataset_64x64 = Dataset(torch.stack(_test_tensor_64x64, dim = 0).unsqueeze(1), _test_labels_64x64)

train_loader_64x64 = DataLoader(train_dataset_64x64, batch_size = batch_size_train, shuffle = True)
test_loader_64x64 = DataLoader(test_dataset_64x64, batch_size = batch_size_test, shuffle = True)

## Import and initialize models
model_52x600 = CNNUntouchedImages().to(device)
model_64x64 = CNNSquareImages().to(device)

# Criterions and optimizers
criterion_52x600 = nn.CrossEntropyLoss()
optimizer_52x600 = optim.Adam(model_52x600.parameters(), lr = learning_rate)

criterion_64x64 = nn.CrossEntropyLoss()
optimizer_64x64 = optim.Adam(model_64x64.parameters(), lr = learning_rate)

## Training session
losses_52x600 = []
losses_64x64 = []

# Train the 52x600 model
print(f'Training the 52x600 CNN')
_time = time.time()
for epoch in range(epochs):
    # print(f'Epoch [{epoch+1}/{epochs}]', end = ' ')
    
    for batch_x, batch_y in train_loader_52x600:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer_52x600.zero_grad()
        scores = model_52x600(batch_x)
        loss = criterion_52x600(scores, batch_y)
        loss.backward()
        optimizer_52x600.step()
    
    losses_52x600.append(loss.item())
    # print(f'Loss : {loss.item()}')

print(f'Training time : {round(time.time() - _time, 2)}s')

# Train the 64x64 model
print(f'Training the 64x64 CNN')
_time = time.time()
for epoch in range(epochs):
    # print(f'Epoch [{epoch+1}/{epochs}]', end = ' ')
    
    for batch_x, batch_y in train_loader_64x64:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer_64x64.zero_grad()
        scores = model_64x64(batch_x)
        loss = criterion_64x64(scores, batch_y)
        loss.backward()
        optimizer_64x64.step()

    losses_64x64.append(loss.item())
    # print(f'Loss : {loss.item()}')

print(f'Training time : {round(time.time() - _time, 2)}s')

## Plot the losses of each model
plt.semilogy([_+1 for _ in range(epochs)], losses_52x600, [_+1 for _ in range(epochs)], losses_64x64)
plt.xlabel('Number of epoch')
plt.ylabel('Error rate')
plt.title('Comparison between the CNN with almost unmodified images (52x600) and square images (64x64)')
plt.legend(['CNN_52x600', 'CNN_64x64'])
plt.show()

## Evaluate the models several times
# Evaluate the 52x600 CNN
model_52x600.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader_52x600:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_52x600(inputs)
        _, predicted = torch.max(outputs, dim = 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
accuracy = 100 * correct / total
print(f'CNN_52x600 accuracy : {accuracy:.2f}')

# Evaluate the 64x64 CNN
model_64x64.eval()
correct, total = 0, 0

with torch.no_grad():
    for inputs, labels in test_loader_64x64:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model_64x64(inputs)
        _, predicted = torch.max(outputs, dim = 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'CNN_64x64 accuracy : {accuracy:.2f}')