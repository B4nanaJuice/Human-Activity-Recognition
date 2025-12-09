import torch.nn as nn
import torch.nn.functional as F

class CNN64x64_3CONV2FC(nn.Module):

    """
    Convolutional Neuronal Network with 3 convolution layers and 2 full connection layers. 
    The CNN takes 64x64 images as an input and output one of the 7 classes.
    """

    def __init__(self, in_channels = 1, num_classes = 7):
        super(CNN64x64_3CONV2FC, self).__init__()

        # 1x64x64
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 1)
        # 16x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        # 32x16x16
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        # 64x8x8

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x