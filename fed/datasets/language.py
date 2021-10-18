import os
import random

import numpy as np
import torch


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus:
    def __init__(self, path='./data'):
        super().__init__()
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids


class WikitextDataset(object):
    def __init__(self, meta, ntokens):
        self.dataset = meta
        self.ntokens = ntokens

    def __iter__(self):
        for inseq, outseq in self.dataset:
            yield (inseq, outseq)
        # yield [(inseq, outseq) for inseq, outseq in self.dataset]

    def __len__(self):
        return len(self.dataset)

    def __delitem__(self, key):
        self.dataset.__delitem__(key)

    def __getitem__(self, key):
        return self.dataset.__getitem__(key)

    def __setitem__(self, key, value):
        self.dataset.__setitem__(key, value)


class WikitextLoader(object):
    def __init__(self, batch_size, meta, sequence_length, mode='train'):
        self.batch_size = batch_size
        self.ntokens = len(meta.dictionary)
        self.train_data = self._batchify(meta.train, batch_size)
        self.val_data = self._batchify(meta.valid, batch_size)
        self.test_data = self._batchify(meta.test, batch_size)
        self.mode = mode
        self.ntokens = len(meta.dictionary)
        self.sequence_length = sequence_length
        self.i = 0

    def _batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    def get_data(self, mode=None):
        if mode is not None:
            self.source = getattr(self, mode+'_data')
        self.source =  getattr(self, self.mode+'_data')
        self.batched_data = []
        for i in range(0, self.source.size(0)-1, self.sequence_length):
            seq_len = min(
                self.sequence_length,
                len(self.source) - 1 - i)
            data = self.source[i:i+seq_len]
            # predict a single word?
            target = self.source[i+1:i+1+seq_len].view(-1)
            self.batched_data.append((data, target))

    def get_batch(self):
        return WikitextDataset(self.batched_data, self.ntokens)


def split(dataset, policy, num_clients, num_shards):
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
    if policy == 'task':
        bins = {}
        for i, (_, label) in enumerate(dataset):
            import pdb; pdb.set_trace()
            bins.setdefault(int(label), []).append(i)
        flattened = []
        for k in sorted(bins):
            flattened += bins[k]
        datasets = []
        if num_shards % num_clients:
            raise ValueError(
                'Expect the number of shards to be '
                'evenly distributed to clients.')
        num_client_shards = num_shards // num_clients
        shard_size = len(dataset) // num_shards
        shards = list(range(num_shards))
        random.shuffle(shards)
        for i in range(num_clients):
            shard_offset = i * num_client_shards
            indices = []
            for s in shards[shard_offset:shard_offset + num_client_shards]:
                if s == len(shards) - 1:
                    indices += flattened[s * shard_size:]
                else:
                    indices += flattened[s * shard_size:(s + 1) * shard_size]
            import pdb; pdb.set_trace()
            subset = torch.utils.data.Subset(dataset, indices)
            datasets.append(subset)
        return datasets
    raise TypeError(f'Unrecognized split policy {policy!r}.')


def Dataset(
        name, train, batch_size, num_clients, num_shards,
        split_mode='task', parallel=False):
    if name == 'wikitext':
        # Finish on the splitting
        corpus = Corpus('./data/wikitext-2')
        loader = WikitextLoader(
            batch_size, corpus, sequence_length=35, mode='train')
        loader.get_data()
        dataset = loader.get_batch()
        # torchtext.datasets.WikiText2('./data',
    if name == 'ag_news':
        dataset = TextClassifyLoader(name, train)
        dataset = dataset.generate_batch()

    if num_clients is None:
        return dataset
    sets = split(dataset, split_mode, num_clients, num_shards)
    return list(sets)


class TextClassifyLoader(object):
    ngrams = 2
    name = 'AG_NEWS'

    def __init__(self, name, train, batch_size=64):
        import torchtext  # FIXME
        name = name.upper()
        dataset_cls = torchtext.datasets.text_classification.DATASETS[name]
        train_dataset, test_dataset = dataset_cls(
            root='./data', ngrams=self.ngrams, vocab=None)
        self.dataset = train_dataset if train else test_dataset
        self.batch_size = batch_size

    def generate_batch(self):
        label = torch.tensor([entry[0] for entry in self.dataset])
        text = [entry[1] for entry in self.dataset]
        offsets = [0] + [len(entry) for entry in text]
        # torch.Tensor.cumsum returns the cumulative sum
        # of elements in the dimension dim.
        # torch.Tensor([1.0, 2.0, 3.0]).cumsum(dim=0)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text = torch.cat(text)
        import pdb; pdb.set_trace()
        return (text, offsets), label
