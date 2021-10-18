from fed.pretty import flops
from fed.models.base import LayerDropout
import sys
import signal
import collections

import torch
from torch.optim import Optimizer
from tblib import pickling_support

from ..pretty import log, get_model_complexity_info, unit
from ..utils import topk, MovingAverage
from ..datasets import INFO, datasets_map
from ..models import factory

class DivergeError(ValueError):
    """ Training loss diverged to NaN.  """


class Process(torch.multiprocessing.Process):
    initialized = False
    len_history = 100

    def __init__(
            self, action, in_queue, out_queue,
            create_func, init_func, loss_func, grad_func,
            model_name, dataset_params, scaling,
            lr, lr_decay_rounds=None, lr_decay_factor=1.0,
            weight_decay=0.0, momentum=0.0,
            parallel=False, device=None, log_level='info'):
        super().__init__()
        self.action = action
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.init_func = init_func
        self.loss_func = loss_func
        self.grad_func = grad_func
        self.model_name = model_name
        self.dataset_params = dataset_params
        self.scaling = scaling
        self.lr = lr
        self.lr_decay_rounds = lr_decay_rounds
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.parallel = parallel
        self.device = device
        self.log_level = log_level
        create_func(self)

    @property
    def id(self):
        return int(self.name.replace("Process-", ""))

    def start(self):
        if self.parallel:
            super().start()

    def terminate(self):
        if self.parallel:
            super().terminate()

    def run(self):
        if not self.parallel:
            raise RuntimeError(
                'This method should only be called by the child process.')
        while True:
            self.call(self.in_queue.get())

    def call(self, info):
        tag, action, kwargs = info
        result = {
            'status': 'ok',
            'tag': tag,
            'client': kwargs.get('client'),
            'process': self.id,
        }
        try:
            if not self.initialized:
                self.init()
            result.update(getattr(self, action)(**kwargs))
        except Exception as e:  # pylint: disable=broad-except
            result.update({
                'status': 'error',
                'exception': e,
            })
        self.out_queue.put(result)
        sys.stdout.flush()

    def init(self):
        if self.initialized:
            raise RuntimeError('Repeated initialization.')
        self.initialized = True
        if self.parallel:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        # model
        dataset_name = self.dataset_params['name']
        info = INFO[dataset_name]
        model_params = info['model_params']
        self.model = factory[self.model_name](
            **model_params, scaling=self.scaling)
        # dataloaders
        self.task = info['task']
        self.ntokens = info.get('ntokens')
        Dataset = datasets_map[dataset_name]
        self.dataloaders = Dataset(**self.dataset_params)
        self.batch_size = self.dataset_params['batch_size']
        # optimizer
        if self.action == 'scaffold':
            self.optimizer = SCAFFOLDOptimizer(
                self.model.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)
        else:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        # self.optimizer = torch.optim.Adam(
        #     self.model.parameters(),
        #     lr=self.lr, amsgrad=True, weight_decay=self.weight_decay)
        # lr decay steps
        self.lr_scheduler = None
        if self.lr_decay_rounds:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.lr_decay_rounds, gamma=self.lr_decay_factor)
            # self.lr = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     self.optimizer, T_max=self.lr_decay_rounds)
        # others
        self.images_seen = 0
        log.level = self.log_level
        # custom
        self.init_func(self)
        self.model = self.model.to(self.device)

    def _get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def _step(self, data, target, init_state, avg_losses, avg_accs):
        data, target = data.to(self.device), target.to(self.device)
        self.optimizer.zero_grad()
        output = self.model(data)
        if self.task == 'language':
            output = output.view(-1, self.ntokens)
        loss = self.loss_func(self, output, target, init_state)
        # import ipdb; ipdb.set_trace()
        if torch.isnan(loss):
            raise DivergeError('Training loss diverged to NaN.')
        avg_losses.add(loss)
        if self.task == 'image':
            train_acc = topk(output, target)[0]
            avg_accs.add(train_acc)
        loss.backward()
        self.grad_func(self, init_state)
        if self.action == 'scaffold':
            self.optimizer.step(self.server_cv, self.client_cv)
        else:
            self.optimizer.step()
        self.images_seen += target.size(0)

    def get_weight(self, client):
        return {'weight': len(self.dataloaders[client].dataset)}

    def _iterate(self, dataloader, steps):
        step = 0
        while True:
            for data, target in dataloader:
                if step >= steps:
                    return
                yield data, target
                step += 1

    def train(self, client, state, steps, rounds, server_cv=None, client_cv=None):
        msg = f'process: {self.id}, ' f'client: {client}'
        avg_accs = MovingAverage(self.len_history)
        avg_losses = MovingAverage(self.len_history)
        self.optimizer.state = collections.defaultdict(dict)
        self.optimizer.zero_grad()
        dataset_size = len(self.dataloaders[client].dataset)
        if self.action == 'scaffold':
            self._init_cv(server_cv, client_cv, client)
        if self.lr_scheduler:
            self.lr_scheduler.last_epoch = rounds - 1
            self.optimizer.zero_grad()
            if self.action == 'scaffold':
                self.optimizer.step(self.server_cv, self.client_cv)
            else:
                self.optimizer.step()  # disable warning on the next line...
            self.lr_scheduler.step()
        self.model.load_state_dict(state)
        flops = self._flops()
        result = {
            'weight': dataset_size,
            'flops.model': flops,
            'flops.total': flops * steps * self.batch_size,
        }
        init_state = {
            k: v.to(self.device, copy=True) for k, v in state.items()}
        self.model.train()
        try:
            for data, target in self._iterate(self.dataloaders[client], steps):
                self._step(data, target, init_state, avg_losses, avg_accs)
        except DivergeError as e:
            log.verbose(f'{msg}, diverged to NaN.')
            return {'status': 'error', 'exception': e, **result}
        log.verbose(
            f'{msg}, train acc: {float(avg_accs.mean()):.2%}, '
            f'lr: {self._get_lr():.3f}, model flops: {unit(flops)}.')
        #sample_flops = self._sample_flops()
        sample_flops=  0.
        result.update({
            'state': {
                k: v.detach().clone()
                for k, v in self.model.state_dict().items()},
            'accuracy': float(avg_accs.mean()),
            'loss': float(avg_losses.mean()),
            'flops.sample': sample_flops,
        })

        if self.action == 'scaffold':
            delta_cv = self._delta_cv()
            result.update({'delta_cv': delta_cv})
            result.update({'client_cv': self.client_cv})
        return result

    def _flops(self):
        input_size = INFO[self.dataset_params['name']]['shape']
        macs = get_model_complexity_info(
            self.model, input_size, as_strings=False,
            # print_per_layer_stat=log.is_enabled('debug'),
            ignore_modules=[torch.nn.Conv2d])[0]
        return macs

    def _sample_flops(self):
        input_size = INFO[self.dataset_params['name']]['shape']
        macs, params, ori_flops = get_model_complexity_info(
            self.model, input_size, as_strings=False,
            # print_per_layer_stat=log.is_enabled('debug'),
            ignore_modules=[torch.nn.Conv2d])
        flops_list = [((), (652288)),
                      ((), (10047744)),
                      ((), (923200)),
                      ((), (136714))]
        probs_list = []
        flops_sum = 0.
        sample_num = 0
        for k, v in self.model.named_modules():
            if isinstance(v, LayerDropout):
                probs = []
                for prob in v.prob_samples:
                    probs.append(prob)
                probs_list.append(probs)
                sample_num = len(v.prob_samples)
                v.prob_samples = []
        def cal_prob(prob):
            numel = prob.numel()
            return prob.sum() / numel

        for p1, p2, p3 in zip(*probs_list):
            p1 = cal_prob(p1)
            p2 = cal_prob(p2)
            p3 = cal_prob(p3)
            flops_sum += p1*flops_list[0][1] + p1*flops_list[1][1]*p2 + p2*flops_list[2][1]*p3 + flops_list[3][1]

        flops_sum *= self.batch_size
        return flops_sum






    def _delta_cv(self,):
        delta_cv = [ccp.data - cc.data for ccp, cc in zip(self.client_cv_plus, self.client_cv)]
        for ccp, cc in zip(self.client_cv_plus, self.client_cv):
            cc.data = ccp.data
        return delta_cv

    def _init_cv(self, server_cv, client_cv, client):
        self.server_cv = server_cv or [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.client_cv = client_cv or [torch.zeros_like(p.data) for p in self.model.parameters() if p.requires_grad]
        self.server_cv = [t.to(self.device) for t in self.server_cv]
        self.client_cv = [t.to(self.device) for t in self.client_cv]
        self.client_cv_plus = self._client_cv_plus(client)

    def _client_cv_plus(self, client):
        self.model.train()
        self.optimizer.zero_grad()
        for i, (data, target) in enumerate(self.dataloaders[client]):
            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            if self.task == 'language':
                output = output.view(-1, self.ntokens)
            loss = torch.nn.functional.cross_entropy(output, target)
            loss.backward()
        cv = [p.data.detach().clone()/(i+1) for p in self.optimizer.param_groups[0]['params']]
        return cv

class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay, cv_lambda=1.):
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
        self.cv_lambda = cv_lambda
        pass

    def step(self, server_controls, client_controls, client=None, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        # Assume there is only one parameter group
        for p, sv, cl in zip(self.param_groups[0]['params'], server_controls, client_controls):
            if p.grad is None:
                continue
            d_p = p.grad.data + self.cv_lambda * (sv.data - cl.data)
            p.data.add_(d_p.data, alpha=-self.param_groups[0]['lr'])
        return loss

pickling_support.install()
