import os
import glob
import pickle
import functools

import torch
import torchvision as tv
import numpy as np
from PIL import Image

from .info import INFO
from .image_preprocess import preprocess_equal_niid, preprocess_random_niid
from .common import combine_data
from ..pretty import log


class Cutout:
    """
    Randomly mask out one or more patches from an image.
    holes: Number of patches to cut out of each image.
    length: The length (in pixels) of each square patch.

    ref: https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    """
    def __init__(self, holes=1, length=16):
        self.holes = holes
        self.length = length

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        mask = np.ones((h, w), np.float32)
        for _ in range(self.holes):
            y = np.random.randint(h)
            x = np.random.randint(w)
            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)
            mask[y1: y2, x1: x2] = 0
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        return img * mask


class TransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        super().__init__()
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        return self.transform(image), label


class ProprocessedDataset(torch.utils.data.Dataset):
    def __init__(self, meta, num_samples):
        super().__init__()
        self.meta_to_np(meta)
        self._num_samples = num_samples

    def meta_to_np(self, meta):
        self.x = [np.array(x).astype(np.float) for x in meta['x']]
        self.y = meta['y']

    def __len__(self):
        return self._num_samples

    def __getitem__(self, index):
        image = self.x[index]
        image = Image.fromarray(image, mode='L')
        return image, int(self.y[index])


def bin_index(dataset):
    bins = {}
    for i, (_, label) in enumerate(dataset):
        bins.setdefault(int(label), []).append(i)
    flattened = []
    for k in sorted(bins):
        flattened += bins[k]
    return bins, flattened


def augment(name, dataset, train):
    augments = []
    flip = cutout = randcrop = train and 'cifar' in name.lower()
    cutout = False
    if randcrop:
        augments.append(tv.transforms.RandomCrop(32, padding=4))
    if flip:
        augments.append(tv.transforms.RandomHorizontalFlip())
    augments.append(tv.transforms.ToTensor())
    if cutout:
        length = {'cifar10': 16, 'cifar100': 8}[name.lower()]
        augments.append(Cutout(1, length))
    if name == "FashionMNIST":
        name = "fashionmnist"
    else:
        name = name.lower()
    info = INFO[name]
    if 'moments' in info:
        augments.append(tv.transforms.Normalize(*info['moments']))
    return TransformDataset(dataset, tv.transforms.Compose(augments))


def split(name, dataset, policy, num_clients, num_shards, alpha):
    # guarantee determinism
    np.random.seed(0)
    torch.manual_seed(0)
    if policy == 'iid':
        splits = [len(dataset) // num_clients] * num_clients
        splits[-1] += len(dataset) % num_clients
        return torch.utils.data.dataset.random_split(dataset, splits)
    if policy == 'size':
        splits = np.random.random(num_clients)
        splits *= len(dataset) / np.sum(splits)
        splits = splits.astype(np.int)
        remains = sum(splits)
        remains = np.random.randint(0, num_clients, len(dataset) - remains)
        for n in range(num_clients):
            splits[n] += sum(remains == n)
        return torch.utils.data.dataset.random_split(dataset, splits.tolist())
    if policy == 'dirichlet':
        min_size = 0
        num_class = INFO[name.lower()]['model_params']['num_classes']
        bins, flattened = bin_index(dataset)
        num_data = len(flattened)
        while min_size < 10:
            idx_batch = [[] for _ in range(num_clients)]
            statistics = {c:[] for c in range(num_clients)}
            for k in range(num_class):
                idx_k = bins[k]
                np.random.shuffle(idx_k)
                prop = np.random.dirichlet(np.repeat(alpha, num_clients))
                prop = np.array([p * (len(idx_c) < num_data / num_clients) for p, idx_c in zip(prop, idx_batch)])
                prop = prop / prop.sum()
                prop = (np.cumsum(prop)*len(idx_k)).astype(int)[:-1]
                for c, (idx_j, idx) in enumerate(zip(idx_batch, np.split(idx_k, prop))):
                    idx_j += idx.tolist()
                    statistics[c].append(len(idx))
                min_size = min([len(idx_c) for idx_c in idx_batch])
        datasets = [torch.utils.data.Subset(dataset, idx_batch[c]) for c in range(num_clients)]
        for c, v in statistics.items():
            log.debug(f'client: {c}, total: {int(np.sum(v))}, data dist: {v}')
        return datasets
    if policy == 'task':
        bins, flattened = bin_index(dataset)
        datasets = []
        if num_shards % num_clients:
            raise ValueError(
                'Expect the number of shards to be '
                'evenly distributed to clients.')
        num_client_shards = num_shards // num_clients
        shard_size = len(dataset) // num_shards
        shards = list(range(num_shards))
        np.random.shuffle(shards)  # fix np.ramdom error
        for i in range(num_clients):
            shard_offset = i * num_client_shards
            indices = []
            for s in shards[shard_offset:shard_offset + num_client_shards]:
                if s == len(shards) - 1:
                    indices += flattened[s * shard_size:]
                else:
                    indices += flattened[s * shard_size:(s + 1) * shard_size]
            subset = torch.utils.data.Subset(dataset, indices)
            datasets.append(subset)
        return datasets
    raise TypeError(f'Unrecognized split policy {policy!r}.')


def check_path(name, split_mode, num_classes, num_clients, np_random_seed):
    if split_mode == "equal_niid":
        fname = 'all_data_equal_niid.pkl'
    elif split_mode == "random_niid":
        fname = "all_data_random_niid.pkl"
    else:
        raise ValueError(f"{split_mode} is not supported!")

    if name == "mnist":
        d_name = "mnist"
    elif name == "cifar10":
        d_name = "cifar10"
    else:
        raise ValueError(f"dataset {name} is not supported!")

    # if data file is not preprocessed, now preprocess
    train_path, eval_path = f'./data/{name}/{d_name}/train/', f'./data/{name}/{d_name}/test/'
    train_file, eval_file = train_path + fname, eval_path + fname
    if (glob.glob(train_file) == [] or glob.glob(eval_file) == []):
        if split_mode == "equal_niid":
            preprocess_cls = preprocess_equal_niid
        elif split_mode == "random_niid":
            preprocess_cls = preprocess_random_niid
        preprocess_cls(
            dataset_name=name,
            num_classes=num_classes,
            num_users=num_clients,
            np_random_seed=np_random_seed,
            download_path=f'./data/{name}')
    else:
        log.info(f'{train_file} and {eval_file} exists.')
    return train_file, eval_file


def Dataset(
        name, train, batch_size, num_clients, num_shards,
        split_mode, parallel=True, alpha=0.5, data_dir=None):
    if split_mode in ['equal_niid', 'random_niid']:
        return NIIDDataset(
            name, split_mode, train, batch_size, num_clients, parallel)
    # dataset creation
    # FIXME messy name config
    if name in ['fmnist', 'fashionmnist', 'fashion_mnist']:
        name = 'FashionMNIST'
    elif name in ['cifar10', 'cifar100', 'mnist', 'svhn', 'emnist']:
        name = name.upper()
    cls = getattr(tv.datasets, name)
    root = data_dir or '../datasets'
    path = os.path.join(root, name.lower())
    kw = {'root': path,
              'train': train}
    if name == 'SVHN':
        kw.pop('train')
        kw['split'] = 'train' if train else 'test'
    if name == 'EMNIST':
        kw['split'] = 'digits'
    try:
        #dataset = cls(path, train=train, download=False)
        dataset = cls(**kw, download=False)
    except RuntimeError:
        #dataset = cls(path, train=train, download=True)
        dataset = cls(**kw, download=True)
    kwargs = {}
    if parallel:
        kwargs = {'pin_memory': False, 'num_workers': 0}
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    if num_clients is None or not train:
        return Loader(
            augment(name, dataset, train), batch_size, train)
    return [
        Loader(augment(name, d, True), batch_size, train)
        for d in split(name, dataset, split_mode, num_clients, num_shards, alpha)]


def NIIDDataset(name, split_mode, train, batch_size, num_clients, parallel):
    num_classes = INFO[name]['model_params']['num_classes']
    # refactor this to cli
    np_random_seed = np.random.seed(6)
    train_file, eval_file = check_path(name, split_mode, num_classes, num_clients, np_random_seed)
    write_out_file = train_file if train else eval_file
    with open(write_out_file, 'rb') as f:
        raw_data = pickle.load(f)
    kwargs = {'pin_memory': True, 'num_workers': 0} if parallel else {}
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    if not train:
        all_data = combine_data(raw_data, num_clients)
        test_set = ProprocessedDataset(all_data['user_data'], all_data['num_samples'])
        return Loader(augment(name, test_set, True, False), batch_size, train)
    datasets = [ProprocessedDataset(raw_data['user_data'][i], raw_data['num_samples'][i]) for i in range(num_clients)]
    return [Loader(augment(name, d, True, False), batch_size, train) for d in datasets]
