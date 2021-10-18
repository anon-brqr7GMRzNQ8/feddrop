import math
import collections

import torch
import numpy as np


use_cuda = torch.cuda.is_available()
torch_cuda = torch.cuda if use_cuda else torch
default_device = torch.device('cuda' if use_cuda else 'cpu')


def sparse_kaiming_(tensor, mode='normal', gain=1.0):
    if tensor.dim() < 2:
        raise ValueError(
            'Fan can not be computed for tensor with fewer than 2 dimensions.')
    fan = tensor[0].numel()
    if fan == 0:
        raise ValueError('Cannot initialize an empty tensor.')
    with torch.no_grad():
        if mode == 'normal':
            std = math.sqrt(2.0 * gain / fan)
            return tensor.normal_(0, std), std
        if mode == 'uniform':
            bound = math.sqrt(6.0 * gain / fan)
            return tensor.uniform_(-bound, bound), bound
        raise ValueError(f'Unrecognized mode {mode}.')


def split_by_sizes(values, sizes):
    if sum(sizes) != len(values):
        raise ValueError(
            'The total slice sizes must equal the length of the list.')
    i = 0
    slices = []
    for s in sizes:
        slices.append((i, i + s))
        i += s
    return [values[s:e] for s, e in slices]


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def normalize(values):
    if isinstance(values, collections.Mapping):
        normed = {k: v / sum(values.values()) for k, v in values.items()}
        return values.__class__(normed)
    values = list(values)
    return [v / sum(values) for v in values]


def dict_gather(mapping, keys=None):
    """
    >>> mapping = {'a': [0, 1, 2], 'b': [3, 4]}
    >>> dict_gather(mapping)
    ([0, 1, 2, 3, 4], {'a': 3, 'b': 2})
    """
    lens = {}
    vs = []
    for k in keys or mapping:
        v = mapping[k].flatten()
        lens[k] = len(v)
        vs.append(v)
    return torch.cat(vs), lens


def dict_scatter(values, key_lens):
    """
    >>> values = [0, 1, 2, 3, 4]
    >>> key_lens = {'a': 3, 'b': 2}
    >>> dict_scatter(values, key_lens)
    {'a': [0, 1, 2], 'b': [3, 4]}
    """
    values = split_by_sizes(values, key_lens.values())
    return dict(zip(key_lens, values))


def dict_diff(before, after):
    return {k: before[k] - after[k] for k in before}


def dict_filter(
        mapping, *, types=None, keys=None,
        prefix=None, suffix=None, prefixes=(), suffixes=()):
    new_mapping = {}
    for k, v in mapping.items():
        if types is not None and not isinstance(v, types):
            continue
        if keys is not None and k not in keys:
            continue
        if prefix is not None and not k.startswith(prefix):
            continue
        if suffix is not None and not k.endswith(suffix):
            continue
        if prefixes and not any(k.startswith(p) for p in prefixes):
            continue
        if suffixes and not any(k.endswith(s) for s in suffixes):
            continue
        new_mapping[k] = v
    return new_mapping


def topk(output, target, k=(1, ), count=False):
    _, pred = output.topk(max(k), 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    batch = 1 if count else target.size(0)
    return [float(correct[:k].sum()) / batch for i, k in enumerate(k)]


def topk_mask(values, k, view):
    threshold = values.topk(k).min()
    return values >= threshold, threshold


def safe_log(value, epsilon=1e-20):
    return torch.log(torch.clamp(value, min=epsilon))


def gumbel(shape, epsilon=1e-20):
    uniform = torch.rand(shape)
    if use_cuda:
        uniform = uniform.cuda()
    return -safe_log(-safe_log(uniform))


def gumbel_softmax(probs_or_logits, temperature=1.0, log=True, epsilon=1e-20):
    logits = safe_log(probs_or_logits, epsilon) if log else probs_or_logits
    output = logits + gumbel(logits.shape)
    return torch.nn.functional.softmax(output / temperature, dim=-1)


def gumbel_topk(probs_or_logits, k, temperature=1.0, log=True, epsilon=1e-20):
    logits = safe_log(probs_or_logits, epsilon) if log else probs_or_logits
    if temperature == 0:
        return torch.topk(logits, k)
    if temperature != 1:
        logits /= temperature
    return torch.topk(logits + gumbel(logits.shape, epsilon), k)


def gumbel_max(probs_or_logits, temperature=1.0, log=True, epsilon=1e-20):
    return gumbel_topk(probs_or_logits, 1, temperature, log, epsilon)


def entropy(output, target, ntokens):
    return [torch.nn.CrossEntropyLoss()(output.view(-1, ntokens), target)]


class AccuracyCounter:
    supported_tasks = ['image', 'language']

    def __init__(self, num, k=(1, ), task='image', ntokens=None):
        super().__init__()
        self.num = num
        self.k = k
        self.correct = [0] * len(k)
        self.entropies = []
        self.size = 0
        if task not in self.supported_tasks:
            raise ValueError(
                f'Task {task!r} not in supprted list {self.supported_tasks}.')
        self.task = task
        self._ntokens = ntokens

    def add(self, output, target):
        self.size += target.size(0)
        if self.task == 'image':
            for i, a in enumerate(topk(output, target, self.k, True)):
                self.correct[i] += a
        if self.task == 'language':
            self.entropies.append(entropy(output, target, self._ntokens))

    def logout(self):
        if self.task == 'image':
            return self.accuracies()
        if self.task == 'language':
            return self.entropy()
        raise ValueError

    def entropy(self):
        return np.mean(self.entropies)

    def accuracies(self):
        for i in range(len(self.k)):
            yield self.correct[i] / self.size

    def errors(self):
        for a in self.accuracies():
            yield 1 - a

    def progress(self):
        return self.size / self.num


class MovingAverage:
    def __init__(self, num):
        super().__init__()
        self.num = num
        self.items = []

    def add(self, value):
        self.items.append(float(value))
        if len(self.items) > self.num:
            self.items = self.items[-self.num:]

    def mean(self):
        return np.mean(self.items)

    def std(self):
        return np.std(self.items)

    def flush(self):
        self.items = []

    def __format__(self, mode):
        text = f'{self.mean():.5f}'
        if 's' not in mode:
            return text
        return text + f'±{self.std() * 100:.2f}%'

    def __float__(self):
        return self.mean()
