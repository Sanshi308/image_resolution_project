import torch
import torch.nn as nn

class SimpleSRCNN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=3):
        super(SimpleSRCNN, self).__init__()
        self.scale_factor = scale_factor

        # SRCNN Layers
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        return x