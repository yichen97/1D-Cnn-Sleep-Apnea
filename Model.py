import numpy
import torch
import torch as torch
from torch import nn, autograd


class CNN_test(nn.Module):
    def __init__(self):
        super(CNN_test, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


class CNN_apnea(nn.Module):
    def __init__(self):
        super(CNN_apnea, self).__init__()
        self.model = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(in_channels=1, out_channels=3,
                      kernel_size=100, stride=2),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),

            nn.Conv1d(in_channels=3, out_channels=50,
                      kernel_size=10, stride=1),
            nn.MaxPool1d(2, stride=2),
            nn.ReLU(),

            nn.Conv1d(in_channels=50, out_channels=30,
                      kernel_size=30, stride=1),
            nn.MaxPool1d(2, 2),
            nn.ReLU(),
            nn.BatchNorm1d(30),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(1950, 2),
            nn.Softmax(1)

        )

    def forward(self, x):
        x = self.model(x)
        return x


if __name__ == '__main__':
    model = CNN_apnea()
    output = model(torch.randn(64, 1, 1408))

