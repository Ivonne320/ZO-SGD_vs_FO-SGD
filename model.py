import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Dropout2d(),
            nn.Linear(50, num_classes),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 320)
        x = self.classifier(x)
        return x