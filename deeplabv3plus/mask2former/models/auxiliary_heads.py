import torch
import torch.nn as nn

class AuxiliaryClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(AuxiliaryClassifier, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.pool(x)          # (B,C,1,1)
        x = torch.flatten(x, 1)   # (B,C)
        return self.fc(x)

