import os
import glob
import random
import pickle
import functools

import numpy as np
import torch

from .common import combine_data


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(
        alpha, beta, num_users=100, num_classes=10, dimension=60, iid=True):
    samples_per_user = np.random.lognormal(4, 2, num_users).astype(int) + 50
    print('>>> Sample per user: {}'.format(samples_per_user.tolist()))

    X_split = [[] for _ in range(num_users)]
    y_split = [[] for _ in range(num_users)]

    # prior for parameters
    mean_W = np.random.normal(0, alpha, num_users)
    mean_b = mean_W
    B = np.random.normal(0, beta, num_users)
    mean_x = np.zeros((num_users, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(num_users):
        if iid == 1:
            mean_x[i] = np.ones(dimension) * B[i]  # all zeros
        else:
            mean_x[i] = np.random.normal(B[i], 1, dimension)

    W_global = b_global = None
    if iid == 1:
        W_global = np.random.normal(0, 1, (dimension, num_classes))
        b_global = np.random.normal(0, 1,  num_classes)

    for i in range(num_users):

        if iid == 1:
            assert W_global is not None and b_global is not None
            W = W_global
            b = b_global
        else:
            W = np.random.normal(mean_W[i], 1, (dimension, num_classes))
            b = np.random.normal(mean_b[i], 1, num_classes)

        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i].extend(xx.tolist())
        y_split[i].extend(yy.tolist())

        # print("{}-th users has {} exampls".format(i, len(y_split[i])))
    return X_split, y_split


def preprocess_synthetic(
        alpha, beta, iid,
        num_classes=10, num_users=100, dimension=60,
        np_random_seed=np.random.seed(6),
        download_path='./data/'):
    dataset_name = 'synthetic_alpha{}_beta{}_{}'.format(alpha, beta, 'iid' if iid else 'niid')
    print('>>> Generate data for {}'.format(dataset_name))
    train_file = f"./data/synthetic/train/{dataset_name}.pkl"
    test_file = f"./data/synthetic/test/{dataset_name}.pkl"

    for file_name in [train_file, test_file]:
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    X, y = generate_synthetic(
        alpha=alpha, beta=beta, iid=iid, 
        num_users=num_users, num_classes=num_classes, dimension=dimension)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}
    
    for i in range(num_users):
        uname = i
        combined = list(zip(X[i], y[i]))
        random.shuffle(combined)
        X[i][:], y[i][:] = zip(*combined)
        num_samples = len(X[i])
        train_len = int(0.9 * num_samples)
        test_len = num_samples - train_len
        
        train_data['users'].append(uname) 
        train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
        train_data['num_samples'].append(train_len)
        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
        test_data['num_samples'].append(test_len)

    with open(train_file, 'wb') as outfile:
        pickle.dump(train_data, outfile, pickle.HIGHEST_PROTOCOL)
    with open(test_file, 'wb') as outfile:
        pickle.dump(test_data, outfile, pickle.HIGHEST_PROTOCOL)


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, meta, num_samples):
        super().__init__()
        self.x, self.y = meta['x'], meta['y']
        self._num_samples = num_samples
        self._x_to_tensor()

    def _x_to_tensor(self):
        self.x = [torch.from_numpy(np.array(x)).float() for x in self.x]
        self.y = [torch.from_numpy(np.array(y)).long() for y in self.y]

    def __len__(self):
        return self._num_samples
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]


name_to_configs = {
    # (alpha, beta, iid)
    'synthetic-niid00': (0, 0, 0),
    'synthetic-niid0p50p5': (0.5, 0.5, 0),
    'synthetic-niid11': (1, 1, 0),
    'synthetic-iid': (0, 0, 1)
}


def Dataset(
        name, train, batch_size, num_clients, num_shards, split_mode, parallel=False, client_parallel=False):
    if not name in  name_to_configs.keys():
        raise ValueError(f'{name} is not a recognised synthetic dataset!')
    alpha, beta, iid = name_to_configs[name]

    # check path
    dataset_name = 'synthetic_alpha{}_beta{}_{}'.format(alpha, beta, 'iid' if iid else 'niid')
    train_file = f"./data/synthetic/train/{dataset_name}.pkl"
    test_file = f"./data/synthetic/test/{dataset_name}.pkl"
    print(glob.glob(train_file) == [] or glob.glob(test_file) == [])
    if (glob.glob(train_file) == [] or glob.glob(test_file) == []):
        preprocess_synthetic(
            alpha=alpha, beta=beta, iid=iid,
            num_classes=10, num_users=num_clients, dimension=60,
            np_random_seed=np.random.seed(6), download_path='./data/')
    with open(train_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(test_file, 'rb') as f:
        test_data = pickle.load(f)

    kwargs = {'pin_memory': True, 'num_workers': 0} if parallel else {}
    Loader = functools.partial(torch.utils.data.DataLoader, **kwargs)
    raw_data = train_data if train else test_data
    if not train:
        all_data = combine_data(raw_data, num_clients)
        test_set = SyntheticDataset(all_data['user_data'], all_data['num_samples'])
        return Loader(test_set, batch_size, train)
    datasets = [SyntheticDataset(raw_data['user_data'][i], raw_data['num_samples'][i]) for i in range(num_clients)]

    return [Loader(d, batch_size, train) for d in datasets]
