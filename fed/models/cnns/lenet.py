import torch
from torch import nn

from ..base import ModelBase


class MLP(ModelBase):
    name = 'mlp'
    input_size = (1, 28, 28)
    filter = 200

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, self.filter),
            nn.ReLU(inplace=True),
            nn.Linear(self.filter, num_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layers(x)
        return x


class LeNet(ModelBase):
    name = 'lenet5'
    input_size = (1, 28, 28)
    filters = [32, 64, 64, 512]

    def __init__(self, num_classes=10,  scaling = 1):
        super().__init__(num_classes)
        f1, f2, f3, f4 = self.filters
        self.features = nn.Sequential(
            nn.Conv2d(self.input_size[0], f1, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(f1, f2, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(f2, f3, 3, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5, True),
            nn.Linear(f3 * 2 * 2, f4),
            nn.ReLU(inplace=True),
            nn.Linear(f4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class SVHNNet(ModelBase):
    name = 'lenet5'
    input_size = (3, 32, 32)
    filters = [32, 64, 64, 512]

    def __init__(self, num_classes=10,  scaling = 1):
        super().__init__(num_classes)
        f1, f2, f3, f4 = self.filters
        self.features = nn.Sequential(
            nn.Conv2d(self.input_size[0], f1, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(f1, f2, 5, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(f2, f3, 3, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5, True),
            nn.Linear(f3 * 2 ** 2, f4),
            nn.ReLU(inplace=True),
            nn.Linear(f4, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
if __name__ == '__main__':
    x = torch.ones(1, 1, 28, 28)
    model = LeNet(10, 1)
    print(model(x))
