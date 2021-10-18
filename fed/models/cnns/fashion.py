from torch import nn
from ..base import ModelBase


class FashionNet(ModelBase):
    # Reference
    # https://github.com/khanguyen1207/My-Machine-Learning-Corner/blob/master/Zalando%20MNIST/fashion.ipynb
    name = 'fashionnet'
    input_size = (1, 28, 28)
    filters = [32, 32, 64, 64, 128]

    def __init__(self, num_classes=10, scaling=1):
        super().__init__(num_classes)
        f1, f2, f3, f4, f5 = self.filters
        self.features = nn.Sequential(
            nn.Conv2d(self.input_size[0], f1, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, 5, 1, 2),
            nn.MaxPool2d(2, 1),
            nn.Dropout(0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f3, f4, 3, 2, 1),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(f4, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
