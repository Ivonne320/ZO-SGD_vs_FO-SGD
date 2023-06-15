import torch
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self, num_classes=10, scale = 1):
        image_size = 28
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(image_size**2, int(128*scale))
        self.fc2 = nn.Linear(int(128*scale), int(64*scale))
        self.fc3 = nn.Linear(int(64*scale), num_classes)
        #add softmax
        self.a = nn.Softmax()
    

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.a(x)
        return x


class ConvNet(nn.Module):
    def __init__(self, num_classes=10, scale=1):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(7*7*64, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
