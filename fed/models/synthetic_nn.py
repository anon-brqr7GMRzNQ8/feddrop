from torch import nn
from .base import ModelBase


class SyntheticNN(ModelBase):
    name = 'synthetic_nn'
    dimension = 60
    def __init__(self, num_classes=10):
        super().__init__(num_classes)
        self.layers = nn.Sequential(
            nn.Linear(self.dimension, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, num_classes),
        )

    def forward(self, x):
        x = self.layers(x)
        return x

