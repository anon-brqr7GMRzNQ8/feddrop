# ref: https://github.com/lx10077/fedavgpy
import os
import pickle

import torch
import torchvision as tv
import numpy as np


class ImageDataset(object):
    def __init__(
            self, images, labels, scale=False, normalize=False,
            mu=None, sigma=None):
        if isinstance(images, torch.Tensor):
            if scale:
                self.data = images.view(-1, 784).numpy()/255
            else:
                self.data = images.numpy()
        else:
            self.data = images
        if normalize:
            self.data = (self.data.astype(np.float32) - mu) / (sigma + 0.001)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)
        self.target = labels

    def __len__(self):
        return len(self.target)


def data_split(data, num_split):
    delta, r = len(data) // num_split, len(data) % num_split
    data_lst = []
    i, used_r = 0, 0
    while i < len(data):
        # consume the remainder
        if used_r < r:
            data_lst.append(data[i:i+delta+1])
            i += delta + 1
            used_r += 1
        else:
            data_lst.append(data[i:i+delta])
            i += delta
    return data_lst


# TODO: adapt this to cifar10 
def choose_two_digit(split_data_lst):
    available_digit = []
    for i, digit in enumerate(split_data_lst):
        if len(digit) > 0:
            available_digit.append(i)
    try:
        lst = np.random.choice(available_digit, 2, replace=False).tolist()
    except:
        lst = available_digit
    return lst


def preprocess_equal_niid(
        dataset_name="mnist", 
        num_classes=10,
        num_users=100, 
        np_random_seed=np.random.seed(6), 
        download_path="./data/"):
    print(f">>> Get {dataset_name} data.")
    if dataset_name == "fmnist":
        cls_name = "FashionMNIST"
    else:
        cls_name = dataset_name.upper()
    dataset_cls = getattr(tv.datasets, cls_name)
    trainset = dataset_cls(download_path, download=True, train=True)
    testset = dataset_cls(download_path, download=True, train=False)

    train_data = ImageDataset(trainset.data, trainset.targets)
    test_data = ImageDataset(testset.data, testset.targets)

    # collect train_data by class indexes
    meta_traindata = []
    for number in range(num_classes):
        idx = train_data.target == number
        meta_traindata.append(train_data.data[idx])
    min_number = min([len(dig) for dig in meta_traindata])
    for number in range(num_classes):
        meta_traindata[number] = meta_traindata[number][:min_number-1]

    split_traindata = []
    for per_class_data in meta_traindata:
        split_traindata.append(data_split(per_class_data, 20))

    meta_testdata = []
    for number in range(num_classes):
        idx = test_data.target == number
        meta_testdata.append(test_data.data[idx])
    split_testdata = []
    for digit in meta_testdata:
        split_testdata.append(data_split(digit, 20))

    data_distribution = np.array([len(v) for v in meta_traindata])
    data_distribution = np.round(data_distribution / data_distribution.sum(), 3)
    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_testdata])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train samples to each user
    train_X = [[] for _ in range(num_users)]
    train_y = [[] for _ in range(num_users)]
    test_X = [[] for _ in range(num_users)]
    test_y = [[] for _ in range(num_users)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(num_users):
        print(user, np.array([len(v) for v in split_traindata]))

        for d in choose_two_digit(split_traindata):
            l = len(split_traindata[d][-1])
            train_X[user] += split_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_testdata[d][-1])
            test_X[user] += split_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()

    # Setup directory for train/test data
    print(f'>>> Set data path for {dataset_name}.')
    train_path = f'{download_path}/{dataset_name}/train/all_data_equal_niid.pkl'
    test_path = f'{download_path}/{dataset_name}/test/all_data_equal_niid.pkl'

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 100 users
    for i in range(num_users):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    # Save user data
    with open(train_path, 'wb') as outfile:
        pickle.dump(train_data, outfile)
    with open(test_path, 'wb') as outfile:
        pickle.dump(test_data, outfile)


def preprocess_random_niid(
        dataset_name="mnist", num_classes=10, num_users=100,
        np_random_seed=np.random.seed(6), download_path="./data/"):
    print(f">>> Get {dataset_name} data.")
    if dataset_name == "fmnist":
        cls_name = "FashionMNIST"
    else:
        cls_name = dataset_name.upper()
    dataset_cls = getattr(tv.datasets, cls_name)
    trainset = dataset_cls(download_path, download=True, train=True)
    testset = dataset_cls(download_path, download=True, train=False)

    train_data = ImageDataset(trainset.data, trainset.targets)
    test_data = ImageDataset(testset.data, testset.targets)

    meta_traindata = []    
    for n in range(num_classes):
        idx = train_data.target == n
        meta_traindata.append(train_data.data[idx])
    split_traindata = []
    for d in meta_traindata:
        split_traindata.append(data_split(d, 20))

    meta_testdata = []
    for n in range(num_classes):
        idx = test_data.target == n
        meta_testdata.append(test_data.data[idx])
    split_testdata = []
    for d in meta_testdata:
        split_testdata.append(data_split(d, 20))
    
    data_distribution = np.array([len(v) for v in meta_traindata])
    data_distribution = np.around(data_distribution / data_distribution.sum(), 3)

    print('>>> Train Number distribution: {}'.format(data_distribution.tolist()))

    digit_count = np.array([len(v) for v in split_traindata])
    print('>>> Each digit in train data is split into: {}'.format(digit_count.tolist()))

    digit_count = np.array([len(v) for v in split_testdata])
    print('>>> Each digit in test data is split into: {}'.format(digit_count.tolist()))

    # Assign train samples to each user
    train_X = [[] for _ in range(num_users)]
    train_y = [[] for _ in range(num_users)]
    test_X = [[] for _ in range(num_users)]
    test_y = [[] for _ in range(num_users)]

    print(">>> Data is non-i.i.d. distributed")
    print(">>> Data is balanced")

    for user in range(num_users):
        print(user, np.array([len(v) for v in split_traindata]))

        for d in choose_two_digit(split_traindata):
            l = len(split_traindata[d][-1])
            train_X[user] += split_traindata[d].pop().tolist()
            train_y[user] += (d * np.ones(l)).tolist()

            l = len(split_testdata[d][-1])
            test_X[user] += split_testdata[d].pop().tolist()
            test_y[user] += (d * np.ones(l)).tolist()

    # Setup directory for train/test data
    print(f'>>> Set data path for {dataset_name}.')
    train_path = f'{download_path}/{dataset_name}/train/all_data_random_niid.pkl'
    test_path = f'{download_path}/{dataset_name}/test/all_data_random_niid.pkl'

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Create data structure
    train_data = {'users': [], 'user_data': {}, 'num_samples': []}
    test_data = {'users': [], 'user_data': {}, 'num_samples': []}

    # Setup 1000 users
    for i in range(num_users):
        uname = i

        train_data['users'].append(uname)
        train_data['user_data'][uname] = {'x': train_X[i], 'y': train_y[i]}
        train_data['num_samples'].append(len(train_X[i]))

        test_data['users'].append(uname)
        test_data['user_data'][uname] = {'x': test_X[i], 'y': test_y[i]}
        test_data['num_samples'].append(len(test_X[i]))

    print('>>> User data distribution: {}'.format(train_data['num_samples']))
    print('>>> Total training size: {}'.format(sum(train_data['num_samples'])))
    print('>>> Total testing size: {}'.format(sum(test_data['num_samples'])))

    with open(train_path, 'wb') as outfile:
        pickle.dump(train_data, outfile)
    with open(test_path, 'wb') as outfile:
        pickle.dump(test_data, outfile)

# for testing
# if __name__ == '__main__':
#     preprocess_equal_niid(dataset_name="cifar10")
#     # preprocess_random_niid()
