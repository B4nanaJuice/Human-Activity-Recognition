import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from dataloader import Dataset

def train_model(model: nn.Module, loader: DataLoader, epochs: int) -> tuple[nn.Module, list[float], float]:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    losses = []
    start_train_time = time.time()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            scores = model(batch_x)
            loss = criterion(scores, batch_y)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
    end_train_time = time.time()
    training_time = round(end_train_time - start_train_time, 2)

    return model, losses, training_time

def evaluate_model(model: nn.Module, dataset: Dataset):
    pass