import torch.nn as nn
import torch.nn.functional as F

class CNNSquareImages(nn.Module):

    def __init__(self, in_channels = 1, num_classes = 7):
        super(CNNSquareImages, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)

        self.fc = nn.Linear(32*16*16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x