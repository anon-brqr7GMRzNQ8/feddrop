import sys
import argparse

import numpy as np
import torch
import ray

from .models import factory
from .sessions import session_map
from .datasets import INFO
from .pretty import log, Summary


_ARGUMENTS = {
    'action': {'type': str, 'help': 'Name of the action to perform.'},
    'dataset': {'type': str, 'help': 'Name of the dataset.'},
    'model': {'type': str, 'help': 'Name of the model.'},
    ('-dd', '--data-dir'): {
        'type': str, 'default': None,
        'help': 'Dataset folder location.',
    },
    ('-cd', '--checkpoint-dir'): {
        'type': str, 'default': None,
        'help': 'Checkpoint name to save/restore.'
    },
    ('-ng', '--num-gpus'): {
        'type': int, 'default': None,
        'help': 'number of gpus for parallel training for clients',
    },
    ('-np', '--num-processes'): {
        'type': int, 'default': 1,
        'help': 'number of gpus for parallel training for clients',
    },
    ('-r', '--resume'): {
        'action': 'store_true',
        'help': 'Resume training from the best checkpoint.',
    },
    ('-v', '--verbose'): {
        'action': 'store_true',
        'help': 'Print verbose messages.',
    },
    ('-vv', '--verbose2'): {
        'action': 'store_true',
        'help': 'Print verbose and debug messages.',
    },
    ('-d', '--debug'): {
        'action': 'store_true',
        'help': 'Create a debugpy server for interactive debugging.',
    },
    '--summary-depth': {
        'type': int, 'default': None,
        'help': 'The depth for recursive summary.',
    },
    '--deterministic': {
        'action': 'store_true',
        'help': 'Deterministic run for reproducibility.',
    },
    ('-rn', '--run-name'): {
        'type': str, 'default': None,
        'help': 'The name of the run.'
    },
    ('-nc', '--num-clients'): {
        'type': int, 'default': 100,
        'help': 'The number of clients.',
    },
    ('-ns', '--num-shards'): {
        'type': int, 'default': 200,
        'help': 'The number of shards to split dataset.',
    },
    ('-sm', '--split-mode', ): {
        'type': str, 'default': 'task',
        'help': 'The policy used to split dataset, supports: iid, size, task.',
    },
    ('-sa', '--split-alpha'): {
        'type': float, 'default': 1,
        'help': 'The alpha for Dirichlet dataset spliting.',
    },
    ('-m', '--max-rounds'): {
        'type': int, 'default': None,
        'help': 'Maximum number of rounds for training.',
    },
    ('-b', '--batch-size'): {
        'type': int, 'default': 10,
        'help': 'Batch size for training.',
    },
    ('-eb', '--eval-batch-size'): {
        'type': int, 'default': 1024,
        'help': 'Batch size for evaluation.',
    },
    ('-mw', '--model-scaling'): {
        'type': float, 'default': 1,
        'help': 'Scales the number of channels in each layer.'
    },
    ('-epr', '--epochs-per-round'): {
        'type': float, 'default': 20,
        'help':
            'Number of local update epochs per federated aggregation round.',
    },
    ('-ee', '--equal-epochs'): {
        'action': 'store_true',
        'help':
            'Use an equal number of local epochs per federated aggregation '
            'round.  Note that the number of steps may not be equal.',
    },
    ('-tf', '--train-fraction'): {
        'type': float, 'default': 0.1,
        'help': 'The proportion of clients to train at each round.',
    },
    ('-lr', '--learning-rate'): {
        'type': float, 'default': 0.1,
        'help': 'Initial learning rate.',
    },
    ('-lrd', '--lr-decay-rounds'): {
        'type': int, 'default': 0,
        'help': 'Number of rounds for each learning rate decay.',
    },
    ('-lrdf', '--lr-decay-factor'): {
        'type': float, 'default': 0.1,
        'help': 'Learning rate decay factor.',
    },
    ('-owd', '--optimizer-weight-decay'): {
        'type': float, 'default': 0,
        'help': 'L2 weight decay.',
    },
    ('-om', '--optimizer-momentum'): {
        'type': float, 'default': 0,
        'help': 'Optimizer momentum.',
    },
    ('-am', '--accel-mode'): {
        'type': str, 'default': 'threshold',
        'choices': [
            'random', 'dense', 'bernoulli', 'bernoulli-mse', 'threshold', 'caldas', 'sparse_upload'],
        'help': 'Accelerate dropout mode. ',
    },
    ('-ab', '--accel-batch'): {
        'action': 'store_true',
        'help':
            'Coarse-grained cross-batch dropouts. '
            'If set, it uses the same dropout samples '
            'across all features in a batch.',
    },
    ('-as', '--accel-scale'): {
        'type': str, 'default': 'train',
        'choices': [
            'train', 'train-mean', 'eval', 'eval-mean', 'eval-cold', 'off'],
        'help':
            'Train-time or eval-time dropout scaling.'
            'If "off", disables all scaling.',
    },
    ('-asm', '--accel-similarity-mode'): {
        'type': str, 'default': 'gradient',
        'choices': ['gradient', 'weighted-gradient'],
        'help':
            'Decides how the [neurons x clients x clients] '
            'similarity tensor is computed.',
    },
    ('-adpl', '--accel-density-per-layer'): {
        'type': float, 'default': None,
        'help':
            'Per layer dropout density constraint, '
            'lower value indicates higher sparsity, range: (0, 1].',
    },
    ('-adpc', '--accel-density-per-client'): {
        'type': float, 'default': None,
        'help':
            'Per device dropout density constraint, '
            'lower value indicates higher sparsity, range: (0, 1].',
    },
    ('-adg', '--accel-density-global'): {
        'type': float, 'default': None,
        'help':
            'Global dropout density constraint, '
            'lower value indicates higher sparsity, range: (0, 1].',
    },
    ('-afpc', '--accel-flops-per-client'): {
        'type': float, 'default': None,
        'help':
            'The fraction of the number of original FLOPs used for each client'
            'in accelerated learning, range: (0, 1]',
    },
    ('-afg', '--accel-flops-global'): {
        'type': float, 'default': None,
        'help':
            'The fraction of the number of overall original FLOPs used '
            'in accelerated learning, range: (0, 1]',
    },
    ('-ar', '--accel-regularizer'): {
        'type': float, 'default': 0.01,
        'help': 'Regularization constant for interior-point method.',
    },
    ('-ag', '--accel-granularity'): {
        'type': int, 'default': None,
        'help': 'The acceleration granularity, range: [0, 4].',
    },
    ('-alr', '--accel-lr', '--accel-learning-rate'): {
        'type': float, 'default': 1.0,
        'help': 'The maximum convex optimization learning rate.',
    },
    ('-ai', '--accel-iterations'): {
        'type': int, 'default': 100_000,
        'help': 'The number of convex optimization iterations.',
    },
    ('-amm', '--accel-momentum'): {
        'type': float, 'default': 0,
        'help': 'The momentum of convex optimization gradient descent.',
    },
    ('-arw', '--accel-reward'): {
        'type': float, 'default': 0,
        'help': 'Reward used in convex optimization.',
    },
    ('-agem', '--accel-gradient-estimate-mode'): {
        'type': str, 'default': 'off',
        'choices': ['train', 'aggregate', 'off'],
        'help': 'Decides where we estimate true gradients.',
    },
    ('-acm', '--accel-constrain-mode'): {
        'type': str, 'default': 'barrier',
        'choices': ['barrier', 'project'],
        'help': 'Decides how we constrain the FLOPs or density of the model.',
    },
    ('-fpmu', '--fedprox-mu'): {
        'type': float, 'default': 0,
        'help': 'FedProx regularization.',
    },
    ('-rt', '--raytune'): {
        'type': str, 'default': None,
        'choices': ['ASHA', 'PBT', 'BOHB'],
        'help': 'Raytune hyperparameter tuning.',
    },
    ('-rr', '--resampling_rounds'): {
        'type': int, 'default': 1,
        'help': 'Resampling rounds for partial paticipation.',
    },
}


def parse():
    p = argparse.ArgumentParser(description='Espeon.')
    for k, v in _ARGUMENTS.items():
        k = [k] if isinstance(k, str) else k
        p.add_argument(*k, **v)
    return p.parse_args()


def _excepthook(etype, evalue, etb):
    # pylint: disable=import-outside-toplevel
    from IPython.core import ultratb
    ultratb.FormattedTB()(etype, evalue, etb)
    for exc in [KeyboardInterrupt, FileNotFoundError]:
        if issubclass(etype, exc):
            sys.exit(-1)
    import ipdb
    ipdb.post_mortem(etb)


def main(args=None, checkpoint_dir=None):
    #torch.multiprocessing.set_sharing_strategy('file_system')
    torch.multiprocessing.set_sharing_strategy('file_descriptor')
    torch.multiprocessing.set_start_method('spawn')
    a = args or parse()

    if a.raytune:
        a.checkpoint_dir = checkpoint_dir
        ray.tune.utils.wait_for_gpu(target_util=0.3)
    if a.verbose:
        log.level = 'verbose'
    elif a.verbose2:
        log.level = 'debug'
    if a.debug:
        # pylint: disable=import-outside-toplevel
        import debugpy
        port = 5678
        debugpy.listen(port)
        log.info(
            'Waiting for debugger client to attach '
            f'to port {port}... [^C Abort]')
        try:
            debugpy.wait_for_client()
            log.info('Debugger client attached.')
        except KeyboardInterrupt:
            log.info('Abort wait.')
            sys.excepthook = _excepthook
    torch.manual_seed(0)
    np.random.seed(0)
    if a.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if a.action in session_map:
        train = session_map[a.action](**vars(a))
        return train.train()
    if a.action == 'info':
        info = INFO[a.dataset]
        model = factory[a.model](
            info['model_params']['num_classes'], scaling=a.model_scaling)
        shape = (a.eval_batch_size, ) + info['shape']
        summary = Summary(model, shape, a.summary_depth)
        return print(summary.format())
    return log.fail_exit(
        f'Unkown action {a.action!r}, accepts: '
        f'info, {", ".join(session_map)}.')
