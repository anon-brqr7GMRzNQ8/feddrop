import os

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import (
    ASHAScheduler, PopulationBasedTraining, HyperBandForBOHB)
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.stopper import TrialPlateauStopper

HYPERPARAMETER_ARGS = {
    'batch_size': tune.choice([4, 8, 16, 32]),
    'optimizer_momentum': tune.choice([0, 0.80, 0.90, 0.95]),
    'learning_rate': tune.choice([0.02, 0.01, 0.005, 0.0025, 0.00125]),
    # 'epochs_per_round': tune.choice([2 ** k for k in range(5)]),
    # 'accel-flops-global': tune.choice([1 / f for f in range(1, 6)])
}



def hyper_config(args):
    from easydict import EasyDict
    for k, v in HYPERPARAMETER_ARGS.items():
        setattr(args, k, v)
    return EasyDict(vars(args))


def hptune(main, args, hpargs=None):
    hpargs = hpargs or HYPERPARAMETER_ARGS
    args = hyper_config(args)
    resources = {'cpu': 20, 'gpu': args.num_gpus}
    max_num_rounds = 100
    num_samples = 32
    perturbation_interval = 20  # for PBT
    distributed_tune = False
    scheduler, alg = None, None
    if distributed_tune:
        # init for multi-server distributed search
        ray.init(
            address='auto',
            _node_ip_address=os.environ['ip_head'].split(':')[0],
            _redis_password=os.environ['redis_password'])
    if args.raytune == 'ASHA':
        scheduler = ASHAScheduler(
            time_attr='rounds', metric='top1', mode='max',
            max_t=max_num_rounds, grace_period=40, reduction_factor=2)
    elif args.raytune == 'PBT':
        scheduler = PopulationBasedTraining(
            time_attr='rounds',
            metric='top1',
            mode='max',
            perturbation_interval=perturbation_interval,
            hyperparam_mutations=hpargs)
    elif args.raytune == 'BOHB':
        alg = TuneBOHB(
            max_concurrent=10, metric='top1', mode='max', seed=0)
        scheduler = HyperBandForBOHB(
            time_attr='rounds', metric='top1', mode='max',
            max_t=max_num_rounds, reduction_factor=2)
    else:
        raise ValueError('Unrecognized scheduler.')
    param_names = {
        'batch_size': 'bs',
        'optimizer_momentum': 'mtm',
        'learning_rate': 'lr',
    }
    reporter = CLIReporter(
        parameter_columns=param_names, metric_columns=['top1', 'rounds'],
        max_report_frequency=10, max_progress_rows=100, metric='top1',
        mode='max')
    trial_stopper = TrialPlateauStopper(
        metric='top1',
        std=0.0001,
        num_results=10,
        grace_period=10,
        mode='max')
    result = tune.run(
        main,
        #local_dir='~/ray_results', name='test_experiment',
        resources_per_trial=resources, config=args, search_alg=alg,
        num_samples=num_samples, scheduler=scheduler,
        progress_reporter=reporter, stop=trial_stopper,
        keep_checkpoints_num=2, # num ckpt for each trial
        checkpoint_score_attr='rounds')
    best_trial = result.get_best_trial('top1', 'max', 'last-10-avg')
    best_args = {
        k: v for k, v in dict(best_trial.config).items() if k in hpargs.keys()}
    print(f'\nBest trial args: {best_args}\n')
