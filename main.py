from dataloader import Dataset
from model_paper import CNN
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

## Import data
_train_tensor = torch.load('tensors/train_tensors_52x600.pt')
_train_labels = torch.load('tensors/train_labels_52x600.pt')
_test_tensor = torch.load('tensors/test_tensors_52x600.pt')
_test_labels = torch.load('tensors/test_labels_52x600.pt')

train_dataset = Dataset(torch.stack(_train_tensor, dim = 0).unsqueeze(1), _train_labels)
test_dataset = Dataset(torch.stack(_test_tensor, dim = 0).unsqueeze(1), _test_labels)

train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = True)

device = "cuda" if torch.cuda.is_available() else "cpu"

## Import model and initialize it

model = CNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

print(f'Training on : {device}')

losses = []
num_epochs = 100
for epoch in range(num_epochs):
    if (epoch+1) % 10 == 0:
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
    if (epoch+1) % 10 == 0:  
        print(f'loss : {loss.item():.4f}')

    # if loss.item() > sum(losses) / len(losses):
    #     break

plt.semilogy(losses)
plt.xlabel('Training epoch')
plt.ylabel('Error rate')
plt.show()

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
