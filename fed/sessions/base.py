import os
import copy
import uuid
import queue
import pprint
import datetime

import torch
from ray import tune

from ..pretty import log, History
from ..utils import default_device, AccuracyCounter
from ..datasets import INFO, datasets_map
from ..models import factory
from .process import DivergeError, Process


class SessionBase:
    len_history = 100

    def __init__(
            self, action, model, dataset, num_clients, num_shards,
            split_mode, split_alpha=0.5, model_scaling=4,
            learning_rate=0.01, lr_decay_rounds=300, lr_decay_factor=0.1,
            optimizer_weight_decay=1e-5, optimizer_momentum=0,
            batch_size=50, eval_batch_size=1024,
            num_gpus=None, num_processes=1,
            resume=False, run_name=None, device=None,
            raytune=None, data_dir=None, checkpoint_dir=None, **kwargs):
        super().__init__()
        if kwargs:
            log.debug(f'Ignored arguments:\n{pprint.pformat(kwargs)}')
        self.hyperparams = ['lr', 'momentum', 'batch_size']
        self.action = action
        self.data_dir = data_dir
        self.dataset_name = dataset
        self.dataset_params = {
            'name': dataset, 'train': True, 'batch_size': batch_size,
            'num_clients': num_clients, 'num_shards': num_shards,
            'split_mode': split_mode, 'alpha': split_alpha,
            'parallel': False, 'data_dir': data_dir,
        }
        self.model_name = model
        self.model_scaling = model_scaling
        self.batch_size = batch_size
        self.lr = learning_rate
        self.lr_decay_rounds = lr_decay_rounds
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = optimizer_weight_decay
        self.momentum = optimizer_momentum
        self.num_clients = num_clients
        self.num_processes = num_processes
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.num_processes = max(
            num_processes, num_processes - num_processes % self.num_gpus)
        self.parallel = self.num_processes > 1
        self.resume = resume
        self.device = device or default_device
        self.state_device = 'cpu'
        self.name = os.path.join(
            self.dataset_name, self.model_name, self.action)
        if run_name:
            self.name = os.path.join(self.name, run_name)
        self._init_model(dataset, model, model_scaling)
        self._init_checkpoint(raytune, checkpoint_dir)
        self._init_dataset(dataset, eval_batch_size, data_dir)
        self._init_clients()

    def _init_model(self, dataset, model, scaling):
        info = INFO[dataset]
        self.input_shape = info['shape']
        model_params = info['model_params']
        self.model = factory[model](**model_params, scaling=scaling)
        self.process_init_func(self)
        self.model = self.model.to(self.device)

    def _init_checkpoint(self, raytune, checkpoint_dir):
        # checkpoint
        self.raytune = raytune
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = os.path.join('models', self.name)
        self.ray_checkpoint = None
        if self.checkpoint_dir:
            self.ray_checkpoint = os.path.join(
                self.checkpoint_dir, 'checkpoint')
        os.makedirs(os.path.split(self.checkpoint)[0], exist_ok=True)
        if self.resume or (self.raytune and self.ray_checkpoint):
            self._init_checkpoint_resume()
        else:
            self._init_checkpoint_fresh()
        # history
        self.tb = History(self.tbname)

    def _init_checkpoint_resume(self):
        checkpoint = self.ray_checkpoint if self.raytune else self.checkpoint
        info = torch.load(checkpoint)
        self.server_state = info['server_state']
        self.states = info['states']
        self.best = info['metric']
        self.metrics = info['metrics']
        self.rounds = info['rounds']
        self.tbname = info['history_name']
        log.info(
            f'Resumed from {checkpoint!r} at {self.rounds} rounds '
            f'with {info["description"]}.')

    def _init_checkpoint_fresh(self):
        self.server_state = {
            k: v.to('cpu', copy=True)
            for k, v in self.model.state_dict().items()}
        self.states = {
            c: copy.deepcopy(self.server_state)
            for c in range(self.num_clients)}
        self.best = None
        self.metrics = {}
        self.rounds = 0
        dtime = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.tbname = os.path.join(self.name, dtime)

    def _init_dataset(self, dataset, eval_batch_size, data_dir):
        info = INFO[dataset]
        self.task = info['task']
        self.ntokens = info.get('ntokens')
        Dataset = datasets_map[dataset]
        self._test_dataloader = Dataset(
            dataset, False, eval_batch_size, None, None, None, True,
            data_dir=data_dir)

    def _init_clients(self):
        qm = torch.multiprocessing if self.parallel else queue
        self._in_queue = qm.Queue()
        self._out_queue = qm.Queue()
        self._processes = []
        for i in range(self.num_processes):
            device = torch.device(
                f'cuda:{i % self.num_gpus}' if self.num_gpus else 'cpu')
            p = Process(
                self.action, self._out_queue, self._in_queue, self.process_create,
                self.process_init_func, self.process_loss_func,
                self.process_grad_func, self.model_name, self.dataset_params,
                self.model_scaling, self.lr, self.lr_decay_rounds,
                self.lr_decay_factor, self.weight_decay, self.momentum,
                self.parallel, device, log.level)
            p.daemon = True
            self._processes.append(p)
        for p in self._processes:
            p.start()
        log.verbose(
            f'Initialized {self.num_processes} process(es) '
            f'on {self.num_gpus} GPU(s).')
        self._async_flags = set()
        client_kwargs = [{'client': c} for c in range(self.num_clients)]
        weights, _ = self.async_call('get_weight', client_kwargs)
        weights = {w['client']: w['weight'] for w in weights}
        self.client_weights = [v for k, v in sorted(weights.items())]

    def eval(self, save=True):
        result = self._eval(self.server_state)
        if self.task == 'image':
            top1, top5 = result
            is_best = self.best is None or top1 > self.best
            info = {
                'metric': top1,
                'metrics': {**self.metrics, 'top1': top1, 'top5': top5},
                'is_best': is_best,
                'description': f'{top1:.2%} top1, {top5:.2%} top5',
            }
            self.tb.add_scalar('eval/top1', top1, self.rounds)
            self.tb.add_scalar('eval/top5', top5, self.rounds)
        elif self.task == 'language':
            is_best = self.best is None or result < self.best
            info = {
                'metric': result,
                'metrics': {**self.metrics, 'entropy': result},
                'is_best': is_best,
                'description': f'loss {result:.3f}',
            }
            self.tb.add_scalar('eval/entropy', result, self.rounds)
        else:
            raise ValueError
        info.update({'history_name': self.tbname})
        self.metrics = info['metrics']
        self.tb.flush()
        text = f'round {self.rounds}: eval accs = {info["description"]}'
        info.update({
            'rounds': self.rounds,
            'states': self.states,
            'server_state': self.server_state,
        })
        if save and self.raytune:
            self.ray_save(info, text)
        '''
        if not info.pop('is_best'):
            log.info(text)
            return info # only save best model
        self.best = info['metric']
        self.best_metrics = info['metrics']
        if save:
            torch.save(info, self.checkpoint)
        log.info(f'{text}, saved')
        '''
        log.info(f'{text}')
        return info

    def ray_save(self, info, text):
        with tune.checkpoint_dir(self.rounds) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint')
            torch.save(info, path)
        log.verbose(f'Saved raytune checkpoint {text!r} at {checkpoint_dir!r}.')
        tune.report(top1=info['metric'], rounds=self.rounds)

    def _eval(self, state=None):
        if state:
            self.model.load_state_dict(state)
        self.model.eval()
        ac = AccuracyCounter(
            len(self._test_dataloader.dataset), (1, 5),
            task=self.task, ntokens=self.ntokens)
        with torch.no_grad():
            for images, labels in self._test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                output = self.model(images)
                ac.add(output, labels)
        return ac.logout()

    def async_call(
            self, method, kwargs, to_raise=(Exception, ), to_ignore=()):
        # call
        for kw in kwargs:
            tag = uuid.uuid4()
            info = (tag, method, kw)
            if self.parallel:
                self._out_queue.put(info)
            else:
                self._processes[0].call(info)
            self._async_flags.add(tag)
        # wait
        results, errors = [], []
        while self._async_flags:
            r = self._in_queue.get()
            self._async_flags.remove(r.pop('tag'))
            if r['status'] == 'ok':
                results.append(r)
                continue
            errors.append(r)
            e = r['exception']
            if any(isinstance(e, i) for i in to_ignore):
                continue
            if any(isinstance(e, r) for r in to_raise):
                raise e
        if not self._in_queue.empty() or not self._out_queue.empty():
            raise RuntimeError('Unexpected clients remain in queue.')
        return results, errors

    def async_train(self, states, steps):
        kwargs = []
        for c, s in states.items():
            kw = {
                'client': c,
                'state': s,
                'steps': steps[c],
                'rounds': self.rounds,
            }
            if self.action == 'scaffold':
                kw.update({'server_cv': self.server_cv,
                          'client_cv': self.client_cvs.get(c)})
            kwargs.append(kw)

        results, errors = self.async_call(
            'train', kwargs, to_ignore=[DivergeError])
        results = {r.pop('client'): r for r in results}
        if errors:
            errors = {e.pop('client') for e in errors}
            nans = ', '.join(str(e) for e in errors)
            log.error(f'Clients {nans} training diverged to NaN.')
        return results, errors

    def process_create(self, process):
        pass

    @staticmethod
    def process_init_func(process):
        pass

    @staticmethod
    def process_loss_func(process, model, output, target, state):
        # default loss function
        # TODO check loss for task == 'language'
        return torch.nn.functional.cross_entropy(output, target)

    @staticmethod
    def process_grad_func(process, init_state):
        pass

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def finalize(self, hparams=None):
        hparams = hparams or self.hyperparams
        hp = {k: getattr(self, k) for k in hparams}
        log.verbose(f'Hyperparameters:\n{pprint.pformat(hp)}')
        log.verbose(f'Metrics:\n{pprint.pformat(self.metrics)}')
        final_metrics = {f'final/{k}': v for k, v in self.metrics.items()}
        self.tb.add_hparams(hp, final_metrics)
        return self.metrics
