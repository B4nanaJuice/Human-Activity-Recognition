import torch.nn as nn
import torch.nn.functional as F

class CNN64x64_2CONV1FC(nn.Module):

    """
    Convolutional Neuronal Network with 2 convolution layers and 1 full connection layer. 
    The CNN takes 64x64 images as an input and output one of the 7 classes.
    """

    def __init__(self, in_channels = 1, num_classes = 7):
        super(CNN64x64_2CONV1FC, self).__init__()

        # 1x64x64
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 1)
        # 16x32x32
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        # 32x16x16

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(32 * 16 * 16, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x