import torch
from torch import nn
from ..base import ModelBase


class MCifarnet(ModelBase):
    name = 'mcifarnet'
    input_size = (3, 32, 32)
    filters = [64, 64, 128, 128, 128, 192, 192, 192]
    kernels = [3, 3, 3, 3, 3, 3, 3, 3]
    strides = [1, 1, 2, 1, 1, 2, 1, 1]
    dropout = [False, False, False, True, False, False, True, False]
    #dropout = [True, True, True, True, True, True, True, True]
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        inputs = 3
        iterer = zip(self.kernels, self.filters, self.strides, self.dropout)
        outputs = None
        layers = []
        for k, outputs, stride, dropout in iterer:
            layers += [
                nn.Conv2d(inputs, outputs, k, stride, 1, bias=False),
                nn.BatchNorm2d(outputs),
                nn.ReLU(inplace=False),
            ]
            if dropout:
                layers.append(nn.Dropout(p=0.5))
            inputs = outputs
        self.layers = nn.Sequential(*layers)
        # classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(outputs, num_classes)

    def forward(self, x):
        x = self.layers(x)
        pooled = self.pool(x)
        return self.fc(pooled.squeeze(-1).squeeze(-1))


class tfCifarnet(ModelBase):
    name = 'tfcifarnet'
    input_size = (3, 24, 24)

    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        inputs = 3
        outputs = None
        features = [
            # conv 1
            nn.Conv2d(3, 64, 5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=False),
            # pool 1
            nn.MaxPool2d(3, 2, padding=1),
            nn.LocalResponseNorm(4),
            # conv 2
            nn.Conv2d(64, 64, 5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=False),
            nn.LocalResponseNorm(4),
            nn.MaxPool2d(3, 2, padding=1),
        ]
        self.features = nn.Sequential(*features)
        fcs = [
            nn.Linear(2304, 384),
            nn.ReLU(inplace=False),
            nn.Linear(384, 192),
            nn.ReLU(inplace=False),
            nn.Linear(192, 10),
        ]
        self.fcs = nn.Sequential(*fcs)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.fcs(x)
