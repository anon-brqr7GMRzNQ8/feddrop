import re
import math
import operator
import itertools
import functools

import torch
import numpy as np

from .sync import Synchronous
from ..pretty import unit, log, get_model_complexity_info
from ..models import LayerDropout
from ..utils import (
    dict_filter, dict_gather, dict_scatter, dict_diff,
    normalize, mean, sparse_kaiming_)


def _keysub_param_to_probs(key):
    return re.sub(r'\.droplayer\.(weight|bias)', '.probs', key)


class Accelerate(Synchronous):
    _hysteresis = 0.90
    _optimize_print_freq = 1000

    def __init__(
            self, *args, accel_mode=None,
            accel_batch=False, accel_scale='train',
            accel_density_per_layer=None,
            accel_density_per_client=None, accel_density_global=None,
            accel_flops_per_client=None, accel_flops_global=None,
            accel_lr=1.0, accel_momentum=0, accel_regularizer=1.0,
            accel_iterations=100_000, accel_resolution=1e-5,
            accel_reward=0, accel_gradient_estimate_mode='train',
            accel_similarity_mode='gradient', accel_constrain_mode='project',
            **kwargs):
        self.accel_mode = accel_mode
        self.accel_batch = accel_batch
        self.accel_scale = accel_scale
        self.accel_density_per_layer = accel_density_per_layer
        self.accel_density_per_client = accel_density_per_client
        self.accel_density_global = accel_density_global
        self.accel_flops_per_client = accel_flops_per_client
        self.accel_flops_global = accel_flops_global
        self.accel_lr = accel_lr
        self.accel_momentum = accel_momentum
        self.accel_regularizer = accel_regularizer
        self.accel_resolution = accel_resolution
        self.accel_iterations = accel_iterations
        self.accel_reward = accel_reward
        self.accel_gradient_estimate_mode = accel_gradient_estimate_mode
        self.accel_similarity_mode = accel_similarity_mode
        self.accel_constrain_mode = accel_constrain_mode
        super().__init__(*args, **kwargs)
        self.hyperparams += [
            'accel_mode',
            'accel_density_per_layer', 'accel_density_per_client',
            'accel_density_global',
            'accel_flops_per_client', 'accel_flops_global',
        ]

    def _init_model_flops_consts(self):
        # a hack to generate FLOPs constants
        probs_dict = dict_filter(self.model.state_dict(), suffix='.probs')
        for p in probs_dict.values():
            if (p != 1).any():
                # please ensure self.model is dense
                raise ValueError('Model is not dense.')
        self.model_flops, _, flops_consts = get_model_complexity_info(
            self.model, self.input_shape, as_strings=False,
            ignore_modules=[torch.nn.Conv2d])
        const = self.model_flops - sum(v for _, v in flops_consts)
        if const < 0:
            raise ValueError('Negative constant FLOPs in the model.')
        flops_consts.append(((), const))
        self.model_flops_consts = flops_consts

    def _uniform_prob(self):
        prob = min(
            1, self.accel_density_global or 1,
            self.accel_density_per_layer or 1,
            self.accel_density_per_client or 1)
        flops_ratio = min(
            1, self.accel_flops_global or 1, self.accel_flops_per_client or 1)
        if flops_ratio >= 1:
            return prob
        # compute initial uniform probabilities to reach intended flops
        # ax^2 + bx + c = 0
        const = lambda order: sum(
            v for ks, v in self.model_flops_consts if len(ks) == order)
        c, b, a = [const(o) for o in range(3)]
        c -= flops_ratio * self.model_flops
        try:
            p = (math.sqrt(b ** 2 - 4 * a * c) - b) / (2 * a)
        except ValueError as e:
            raise ValueError('Target FLOPs reduction is unattainable.') from e
        return min(prob, p)

    def _init_params(self, model, gain):
        stds = {}
        for kv, v in model.named_parameters():
            if kv == 'classifier.0.weight':
                # FIXME a hack to init FC with a sparse input
                sparse_kaiming_(v, 'uniform', gain)
            kp = _keysub_param_to_probs(kv)
            if kp == kv or kv.endswith('.droplayer.bias'):
                continue
            # input channels are dense
            sparse_kaiming_(v, 'uniform', gain)
        return stds

    def _init_checkpoint(self, raytune, checkpoint_dir):
        self._init_model_flops_consts()
        gain = self._uniform_prob()
        # if self.accel_scale == 'eval-cold':
        #     gain = gain ** -0.5
        self._init_params(self.model, gain)
        super()._init_checkpoint(raytune, checkpoint_dir)

    def _init_checkpoint_fresh(self):
        super()._init_checkpoint_fresh()
        # random probs start
        p = 1 if self.accel_mode == 'dense' else self._uniform_prob()
        self.states = {
            c: self._set_uniform_probs(s, p)
            for c, s in self.states.items()}
        self._set_uniform_probs(self.server_state, p)
        log.verbose(
            f'Initialized models with a constant dropout probability {p:.3%}.')

    def process_create(self, process):
        super().process_create(process)
        process.accel_mode = self.accel_mode
        process.accel_batch = self.accel_batch
        process.accel_scale = self.accel_scale
        process.accel_pgrad = self.accel_gradient_estimate_mode == 'train'

    @staticmethod
    def process_init_func(process):
        func = functools.partial(
            LayerDropout.append_dropout, accel_mode=process.accel_mode,
            accel_batch=process.accel_batch, accel_scale=process.accel_scale)
        process.model.replace_module(func)

    @staticmethod
    def process_grad_func(process, init_state):
        if process.accel_pgrad:
            return
        # reduce the magnitude of the gradient to make training less volatile
        for kv, v in process.model.named_parameters():
            if not v.requires_grad:
                continue
            kp = re.sub(r'\.droplayer\.(weight|bias)', '.probs', kv)
            if kp == kv:
                continue
            p = init_state[kp]
            v.grad *= p.reshape(*p.shape, *([1] * (v.ndim - p.ndim)))

    def _set_uniform_probs(self, state, prob):
        for k, v in state.items():
            if not k.endswith('.probs'):
                continue
            if self.accel_mode == 'caldas':
                num = v.shape[0]
                size = int(num * prob)
                idx = np.random.choice(num, size=size, replace=False)
                state[k] = torch.zeros_like(v)
                state[k][idx] = 1.
            else:
                state[k] = torch.ones_like(v) * prob
        return state

    def _diff_states(self, avg_state, states):
        diff_states = {}
        for c, state in states.items():
            diff_states[c] = dict_diff(avg_state, state)
        return diff_states

    def _states_to_probs(self, state_dicts):
        probs = []
        keylens = []
        for s in state_dicts.values():
            cp = dict_filter(s, suffix='.probs')
            cp, kl = dict_gather(cp)
            probs.append(cp)
            keylens.append(kl)
        # neurons x clients
        return torch.stack(probs, 1), keylens

    def _probs_to_states(self, clients, probs, probs_keylens):
        probs = probs.permute(1, 0)  # clients x neurons
        states = {}
        for k, p, l in zip(clients, probs, probs_keylens):
            states[k] = dict_scatter(p, l)
        return states

    def _covariance_and_rewards(self, diff_states, weights):
        client_ids = list(diff_states.keys())
        num_clients = len(client_ids)
        # we consider only ".weight"s to compute covariance
        avg_state = dict_filter(self.server_state, suffix='.droplayer.weight')
        key_lens = {k: avg_state[k].size(0) for k in avg_state}
        num_neurons = sum(key_lens.values())
        # covariance
        sim = torch.zeros([num_clients, num_clients, num_neurons])
        iterer = itertools.product(range(num_clients), range(num_clients))
        for i, j in iterer:
            dps = []
            ci, cj = client_ids[i], client_ids[j]
            di, dj = diff_states[ci], diff_states[cj]
            # dot product
            for k, v in avg_state.items():
                l = key_lens[k]
                dik = weights[ci] * di[k].reshape(l, -1)
                djk = weights[cj] * dj[k].reshape(l, -1)
                dp = dik * djk
                if self.accel_similarity_mode == 'weighted-gradient':
                    dp *= v.reshape(l, -1) ** 2
                dps.append(dp.sum(1))
            sim[i, j] = torch.cat(dps)
        sim = sim.permute(2, 0, 1).contiguous()  # neurons x clients x clients
        # rewards
        rewards = torch.zeros([num_clients, num_neurons])
        for i in range(num_clients):
            c = client_ids[i]
            d = diff_states[c]
            r = []
            for k, v in avg_state.items():
                l = key_lens[k]
                r.append(weights[c] * (d[k] * v).reshape(l, -1).sum(1))
            rewards[i] = torch.cat(r)
        rewards = rewards.transpose(0, 1)  # neurons x clients
        return sim, rewards

    def _accel_objective(self, sim, rewards, x=None):
        if x is None:
            if self.accel_mode == 'bernoulli-mse':
                return 0, 0
            return sim.sum(), -rewards.sum()
        x_l = x.unsqueeze(1)
        x_r = x.unsqueeze(2)
        if self.accel_mode in ['bernoulli', 'bernoulli-mse']:
            cov = (x_l @ sim @ x_r).sum()
        elif self.accel_mode in ['threshold', 'sparse_upload']:
            cov = (sim * torch.minimum(x_l, x_r)).sum()
        else:
            raise ValueError('Unrecognized accel_mode.')
        reward = -(self._accel_probs_finalize(x) * rewards).sum()
        return cov, reward

    def _accel_sim_init(self, sim, probs):
        probs_t = probs.unsqueeze(1)  # neurons x 1 x clients
        probs = probs.unsqueeze(2)  # neurons x clients x 1
        # $ C^u \get \frac{\hat{C}^u}{\pi^u \circ {\pi^u}^\top} $
        if self.accel_mode in ['bernoulli', 'bernoulli-mse', ]:
            return sim * probs * probs_t
        if self.accel_mode in ['threshold','sparse_upload']:
            return sim * torch.maximum(probs, probs_t)
        raise ValueError

    def _accel_flops(self, probs, keylens):
        flops = 0
        probs_dict = dict_scatter(probs, keylens)
        for keys, const in self.model_flops_consts:
            terms = (const, *(probs_dict[k].clamp(0, 1).mean(0) for k in keys))
            flops += functools.reduce(operator.mul, terms)
        return flops

    def _accel_regularize(self, x, keylens, weights):
        # interior point regularizer
        # x: neurons x clients
        # keylens: Dict[str, int]
        # weights: clients
        if self.accel_mode in ['bernoulli', 'threshold', 'sparse_upload']:
            p = 1.0 / x
        elif self.accel_mode == 'bernoulli-mse':
            p = 1.0 / (x + 1)
        else:
            raise ValueError
        def barrier(value, max_value):
            return torch.log(max_value - value)
        # probs > 0
        reg = [-1 / p.mean()]
        # densities
        d = self.accel_density_per_layer or 1
        if d < 1:
            ps = p.split(keylens.values(), 0)
            reg.append(mean(barrier(s.mean(0), d).mean() for s in ps))
        d = self.accel_density_per_client or 1
        if d < 1:
            reg.append(barrier(p.mean(0), d).mean())
        d = self.accel_density_global or 1
        if d < 1:
            # higher penalty for clients with more local steps
            reg.append(barrier((p * weights.unsqueeze(0)).sum(1).mean(), d))
        # flops
        fpc = self.accel_flops_per_client or 1
        fg = self.accel_flops_global or 1
        #fgs = self._accel_optimize_flops_schedule()
        #fg = fgs['global']
        r = self._accel_flops(p, keylens) / self.model_flops
        if fpc < 1:
            reg.append(barrier(r, fpc).mean())
        if fg < 1:
            reg.append(barrier((r * weights).sum(), fg))
        return -sum(reg), r

    def _accel_probs_to_x(self, probs):
        x = 1.0 / probs
        if self.accel_mode == 'bernoulli-mse':
            x -= 1  # x = 1 / probability - 1
        return x

    def _accel_solution_init(self, probs, keylens, weights):
        x = self._accel_probs_to_x(probs)
        if self.accel_constrain_mode == 'barrier':
            reg, _ = self._accel_regularize(x, keylens, weights)
            if reg.isnan() or reg.isinf():
                # initial solution
                log.verbose(
                    'Initial probabilities violate optimization constraints, '
                    'set to initialization.')
                ones = torch.ones_like(x)
                # ensure constraints differentiable
                x = ones / (self._uniform_prob() * self._hysteresis)
        elif self.accel_constrain_mode == 'project':
            x, _ = self._accel_project(x, keylens)
        else:
            raise ValueError
        return x

    def _accel_project(self, x, keylens):
        if self.accel_constrain_mode == 'project':
            p = self._accel_probs_finalize(x)
            k = torch.arange(
                0, 1 / self._uniform_prob(), 0.001, device=p.device)
            # neurons x clients x factors
            sp = p.unsqueeze(2) * k.reshape(1, 1, -1)
            # clients x factors
            r = self._accel_flops(sp, keylens) / self.model_flops
            i = ((r.mean(0) - self.accel_flops_global) > 0).nonzero()
            if i.numel() == 0:
                p = torch.ones_like(p)
                fr = 1
            else:
                p = (p * k[i.min()]).clamp(0.0, 1.0)
                fr = r[:, i.min()]
            return self._accel_probs_to_x(p), fr
        if self.accel_mode in ['bernoulli', 'threshold', 'sparse_upload']:
            return torch.clamp(x, min=1.0), None
        if self.accel_mode == 'bernoulli-mse':
            return torch.clamp(x, min=0.0), None
        raise ValueError

    def _accel_converge_rate(self, loss, last_loss):
        return 1 if last_loss is None else (last_loss - loss) / abs(last_loss)

    def _accel_probs_finalize(self, x):
        if self.accel_mode in ['bernoulli', 'threshold', 'sparse_upload']:
            return 1 / x
        if self.accel_mode == 'bernoulli-mse':
            return 1 / (x + 1)
        raise ValueError

    def _accel_progress(
            self, i, probs, fr, cov, nodrop_cov, reward, nodrop_reward, loss):
        if self.accel_mode in ['bernoulli', 'threshold']:
            cov_info = f'Δcov: {1 - cov / nodrop_cov:.2%}'
        else:
            cov_info = f'cov: {cov}'
        probs_mean = probs.mean(0)
        probs_range = (probs_mean.max() - probs_mean.min()) / 2
        info = [
            f'round {self.rounds}',
            f'optimize: {unit(i + 1, asint=True, base=1000)}',
            cov_info, f'reward: {reward:.3g}',
            f'density: {probs.mean():.2%}±{probs_range:.2%}',
            f'speedup: {1 / fr.mean():.2f}x',
            f'loss: {loss:.3g}',
        ]
        return info

    def _accel_solution_optimize(self, sim, rewards, x_init, keylens, weights):
        lr = self.accel_lr
        mmt = self.accel_momentum
        res = self.accel_resolution
        reg = self.accel_regularizer
        rwd = self.accel_reward
        x = x_prev = x_init
        loss = loss_last = cov_last = None
        sim_mag = sim.sum()
        sim /= sim_mag
        rewards_mag = rewards.norm()
        rewards /= rewards_mag
        nodrop_cov, nodrop_reward = self._accel_objective(sim, rewards)
        grad_moment = 0
        for i in range(self.accel_iterations):
            if x.isnan().any() or x.isinf().any():
                log.error('Diverged, resetting to initialization...')
                x = x_init
            x_var = torch.autograd.Variable(x, requires_grad=True)
            cov, reward = self._accel_objective(sim, rewards, x_var)
            cov = torch.max(nodrop_cov, cov)
            barrier, fr = 0, None
            if self.accel_constrain_mode == 'barrier':
                barrier, fr = self._accel_regularize(x_var, keylens, weights)
                if barrier.isnan():
                    x, lr = x_prev, lr / 2
                    grad_moment = 0
                    if lr < self.accel_lr * 2 ** -8:
                        log.fail('Unable to find a small enough LR that works.')
                    log.warn(
                        'Optimization constraints violated, resetting to '
                        f'previous value and adjusting LR to {lr}...')
                    continue
            loss = cov + rwd * reward
            grad = torch.autograd.grad(loss + reg * barrier, x_var)[0]
            if mmt > 0:
                update = grad_moment = mmt * grad_moment + lr * grad
            else:
                update = lr * grad
            x_prev, (x, pfr) = x, self._accel_project(x - update, keylens)
            fr = pfr if fr is None else fr
            if (i + 1) % self._optimize_print_freq:
                continue
            # covariance without dropout
            rate = self._accel_converge_rate(loss, loss_last)
            probs = self._accel_probs_finalize(x)
            info = self._accel_progress(
                i, probs, fr, cov, nodrop_cov, reward, nodrop_reward, loss)
            log.info(f'{", ".join(info)}.', update=True)
            if rate <= res:
                log.verbose(
                    f'Early exit from optimization at {i + 1} '
                    f'iterations(s) with convergence rate {rate:.5g}.')
                break
            cov_last = min(cov, cov_last or cov)
            loss_last = min(loss, loss_last or loss)
        info = {
            'loss': loss,
            'cov': cov * sim_mag,
            'nodrop_cov': nodrop_cov * sim_mag,
            'reward': reward * rewards_mag,
            'nodrop_reward': nodrop_reward * rewards_mag,
            'constraints': barrier,
        }
        self.tb.add_multiple_scalars('aggregate/optimize', info, self.rounds)
        return self._accel_probs_finalize(x)

    def _accel_optimize(self, sim, rewards, probs, keylens, weights):
        """
        shapes:
          - sim: neurons x clients x clients
          - rewards: neurons x clients
          - probs: neurons x clients
          - keylens: sum(keylens) == neurons
          - weights: clients
        """
        sim, rewards, probs, weights = [
            v.to(self.device) for v in (sim, rewards, probs, weights)]
        sim = self._accel_sim_init(sim, probs)
        # normalize similarity matrix
        min_evs = [s.eig().eigenvalues.min() for s in sim]
        min_evs = torch.tensor(min_evs, device=self.device)
        if min_evs.min() < 0:
            log.verbose(
                'Adjusting similarity matrix as it is not positive '
                f'semidefinite, minimum eigenvalue = {min_evs.min():.5g}.')
            eye = torch.eye(probs.size(1), device=self.device).unsqueeze(0)
            min_evs = torch.clamp(min_evs, max=0)
            sim_diff = eye * min_evs.reshape(min_evs.size(0), 1, 1)
            sim -= sim_diff
        # initial solution
        x_init = self._accel_solution_init(probs, keylens, weights)
        # optimize
        probs = self._accel_solution_optimize(
            sim, rewards, x_init, keylens, weights)
        return probs.cpu()

    def _average_state_diff(self, diff_states, prob_states, weights):
        if self.accel_gradient_estimate_mode != 'aggregate':
            return self._average(diff_states, weights=weights)
        for c, s in diff_states.items():
            for kv in s:
                kp = re.sub(r'\.droplayer\.(weight|bias)', '.probs', kv)
                if kp == kv:
                    continue
                p = prob_states[c][kp]
                s[kv] *= p.reshape(*p.shape, *([1] * (s[kv].ndim - p.ndim)))
        return self._average(diff_states, weights=weights)

    def aggregate(self, server_state, states, errors, next_clients):
        # TODO RNNs
        # FIXME properly support train_fraction
        # if set(next_clients) != set(states):
        #     raise NotImplementedError
        rand_state = dict_filter(states[list(states)[0]], suffix='.rand_state')
        if self.accel_mode in ['random', 'dense', 'caldas']:
            avg_state = self._average(states, errors)
            avg_state.update(rand_state)
            p = self._uniform_prob() if self.accel_mode in ['random', 'caldas'] else 1
            self._set_uniform_probs(avg_state, p)
            return avg_state, self._duplicate(avg_state, next_clients)
        # similarity matrix
        diff_states = self._diff_states(self.server_state, states)
        weights = normalize({c: self.client_weights[c] for c in states})
        sim, rewards = self._covariance_and_rewards(diff_states, weights)
        # optimization
        probs, keylens = self._states_to_probs(states)
        if self.equal_epochs:
            prob_weights = torch.tensor(list(weights.values()))
        else:
            # equal weights on neurons, if training steps are equal
            prob_weights = torch.ones(len(weights)) / len(weights)
        probs = self._accel_optimize(
            sim, rewards, probs, keylens[0], prob_weights)
        self.tb.add_histogram('aggregate/probs', probs.flatten(), self.rounds)
        # aggregate
        prob_states = self._probs_to_states(diff_states.keys(), probs, keylens)
        avg_diff_state = self._average_state_diff(
            diff_states, prob_states, weights)
        avg_state = {k: server_state[k] - d for k, d in avg_diff_state.items()}
        # dispatch
        update_states = self._duplicate(avg_state, next_clients)
        for c, s in update_states.items():
            s.update(rand_state)
            if c in errors or c not in prob_states:
                # not in previously trained clients,
                # re-cycle last probabilities
                p = dict_filter(self.states[c], suffix='.probs')
                if c in errors:
                    # error running client, reset probs to random
                    p = self._set_uniform_probs(p, self._uniform_prob())
            else:
                p = prob_states[c]
            s.update(p)
        return avg_state, update_states
