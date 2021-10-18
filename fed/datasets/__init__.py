from .image import Dataset as ImageDataset
from .language import Dataset as LanguageDataset
from .synthetic import Dataset as SyntheticDataset
from .info import INFO


datasets_map = {
    'synthetic-niid00': SyntheticDataset,
    'synthetic-niid0p50p5': SyntheticDataset,
    'synthetic-niid11': SyntheticDataset,
    'synthetic-iid': SyntheticDataset,
    'mnist-equal-niid': ImageDataset,
    'mnist-random-niid': ImageDataset,
    'mnist': ImageDataset,
    'emnist': ImageDataset,
    'fmnist': ImageDataset,
    'fashionmnist': ImageDataset,
    'cifar10': ImageDataset,
    'cifar100': ImageDataset,
    'cifar10tf': ImageDataset,
    'svhn': ImageDataset,
    'wikitext': LanguageDataset,
    'ag_news': LanguageDataset,
}
