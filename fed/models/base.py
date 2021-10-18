import torch
from torch import nn


class ModelBase(nn.Module):
    name = None
    input_size = None

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def replace_module(self, replace_func, model=None, prefix=None):
        if model is None:
            model = self
        for name, child in model.named_children():
            rname = name if prefix is None else f'{prefix}.{name}'
            replacement = replace_func(name=rname, module=child)
            if replacement is not None:
                setattr(model, name, replacement)
            else:
                self.replace_module(replace_func, child, rname)


class LayerDropout(nn.Module):
    @staticmethod
    def append_dropout(name, module, accel_mode, accel_batch, accel_scale):
        if not isinstance(module, torch.nn.Conv2d):
            return None
        return LayerDropout(module, accel_mode, accel_batch, accel_scale)

    def __init__(self, layer, mode, batch, scale):
        super().__init__()
        self.droplayer = layer
        if not isinstance(layer, nn.Conv2d):
            raise NotImplementedError
        ones = torch.ones(layer.out_channels)
        self.probs = nn.Parameter(ones, requires_grad=False)
        self.rand = torch.Generator(device='cpu')
        self.rand_state = nn.Parameter(
            self.rand.get_state(), requires_grad=False)
        self.mode = mode
        self.batch = batch
        self.scale = scale
        self.prob_samples = []

    def sample(self, probs):
        if self.mode == 'dense':
            return torch.ones_like(probs)
        if self.mode in ['random', 'bernoulli', 'bernoulli-mse']:
            return torch.bernoulli(probs)
        if self.mode in ['threshold', 'sparse_upload']:
            # FIXME hack for rand_state sync
            self.rand.set_state(self.rand_state.cpu())
            r = torch.rand(probs.shape, generator=self.rand, device='cpu')
            self.prob_samples.append((probs >= r.to(probs.device)).to('cpu').float())
            self.rand_state.copy_(self.rand.get_state())
            return (probs >= r.to(probs.device)).float()
        if self.mode == 'caldas':
            return probs
        raise ValueError('Unrecognized mode.')

    def forward(self, x):
        probs = self.probs.reshape(1, -1, 1, 1)
        scale = probs.mean() if 'mean' in self.scale else probs
        x = self.droplayer(x)
        if self.training:
            if self.batch:
                probs = probs.tile(x.size(0), 1, 1, 1)
            b = self.sample(probs)
            if 'train' in self.scale or 'eval-cold' in self.scale:
                if self.mode == 'caldas':
                    pass
                else:
                    b /= scale
            # print(f'{x.shape} {(x * b).std()}')
            return x * b
        if 'eval-cold' in self.scale:
            b = (probs >= 0.5) / scale
            return x * b
        if 'eval' in self.scale:
            x = x / scale
        if self.mode == 'caldas':
            x = x * scale.mean()
        return x
