import torch.nn as nn
import torch.nn.functional as F

class CNN52x600_3CONV1FC(nn.Module):

    """
    Convolutional Neuronal Network with 3 convolution layers and 1 full connection layer. 
    The CNN takes 52x600 images as an input and output one of the 7 classes.
    """

    def __init__(self, in_channels = 1, num_classes = 7):
        super(CNN52x600_3CONV1FC, self).__init__()

        # 1x52x600
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size = 3, padding = 1)
        # 16x26x300
        self.conv2 = nn.Conv2d(16, 32, kernel_size = 3, padding = 1)
        # 32x13x150
        self.conv3 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        # 64x6x75

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(64 * 6 * 75, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x