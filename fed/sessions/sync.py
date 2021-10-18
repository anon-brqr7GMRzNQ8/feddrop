import math
import time
import random

import torch
import numpy as np
import copy

from ..pretty import log, unit
from ..utils import normalize
from .base import SessionBase
from .process import DivergeError


class Synchronous(SessionBase):
    def __init__(
            self, *args, max_rounds=None,
            epochs_per_round=20, equal_epochs=False,
            train_fraction=0.1, fedprox_mu=0, resampling_rounds, **kwargs):
        self.fedprox_mu = fedprox_mu
        self.equal_epochs = equal_epochs
        self.max_rounds = max_rounds
        self.epochs_per_round = epochs_per_round
        self.train_fraction = train_fraction
        self.resampling_rounds = resampling_rounds
        super().__init__(*args, **kwargs)
        self.hyperparams += [
            'max_rounds', 'epochs_per_round', 'equal_epochs', 'fedprox_mu']
        self._client_schedule = None

    def process_create(self, process):
        process.equal_epochs = self.equal_epochs
        process.fedprox_mu = self.fedprox_mu

    @staticmethod
    def process_loss_func(process, output, target, state):
        loss = torch.nn.functional.cross_entropy(output, target)
        if process.fedprox_mu <= 0:
            return loss
        ploss = 0
        for n, p in process.model.named_parameters():
            if p.requires_grad:
                ploss += torch.nn.functional.mse_loss(
                    p, state[n], reduction='sum')
        return loss + 0.5 * process.fedprox_mu * ploss

    def _dispense_clients(self, count=None, clients=None, strategy='uniform'):
        """ uniform: subsample from new client set
            partial: detach r old clients & add r new clients
        """
        # return sorted(random.choices(
        #     range(self.num_clients), k=num_train_clients))
        if count is None:
            count = max(1, int(self.num_clients * self.train_fraction))
        out = []
        old_clients = copy.deepcopy(clients)
        for _ in range(count):
            if not self._client_schedule:
                self._client_schedule = list(range(self.num_clients))
                random.shuffle(self._client_schedule)
            if strategy == 'uniform':
                out.append(self._client_schedule.pop())
            elif strategy == 'partial':
                out.append(self._client_schedule.pop())
                random.shuffle(old_clients)
                old_clients.pop()
        if strategy == 'partial':
            out += old_clients

        return sorted(out)

    def _round(self, rounds, clients, next_clients, steps):
        # train
        states = {c: s for c, s in self.states.items() if c in clients}
        results, errors = self.async_train(states, steps)
        if not results:
            raise DivergeError('All clients trained to divergence.')
        # aggregate
        states, losses = {}, []
        for c, rv in results.items():
            states[c] = {
                k: v.to(self.state_device) for k, v in rv['state'].items()}
            if self.action == 'scaffold':
                self.delta_cv = {} # only collect delta_cv from trained clients
                self.delta_cv[c] = [cv.cpu() for cv in rv['delta_cv']]
                self.client_cvs[c] = [cv.cpu() for cv in rv['client_cv']]
            if rv['weight'] != self.client_weights[c]:
                raise RuntimeError('Potential dataset split error.')
            losses.append(rv['loss'])
        avg_loss = np.mean(losses)
        # comms
        comms = self.communication_cost(clients, next_clients, states)
        avg_comms = comms * 2 / (len(clients) + len(next_clients))
        self.metrics['comms'] = self.metrics.get('comms', 0) + comms
        self.tb.add_scalar('train/comms/round/average', avg_comms, rounds)
        self.tb.add_scalar('train/comms/total', self.metrics['comms'], rounds)
        self.server_state, update_states = self.aggregate(
            self.server_state, states, errors, next_clients)
        self.states.update(update_states)
        # info
        self.tb.add_scalar('train/nans', len(errors), rounds)
        # loss
        self.tb.add_scalar('train/loss', avg_loss, rounds)
        # flops
        flops = self.flops(results)
        round_flops = flops['total']
        self.metrics['flops'] = self.metrics.get('flops', 0) + round_flops
        self.tb.add_scalar('train/flops/total', self.metrics['flops'], rounds)
        client_flops = np.array(list(flops['step.clients'].values()))
        self.tb.add_histogram('train/flops/step/clients', client_flops, rounds)
        avg_flops = flops['step.average']
        self.tb.add_scalar('train/flops/step/average', avg_flops, rounds)
        sample_flops = flops['flops.sample']
        self.metrics['flops.sample'] = self.metrics.get('flops.sample', 0) + sample_flops
        self.tb.add_scalar('train/flops/total/sampling', self.metrics['flops.sample'], rounds)
        # progress
        info = (
            f'round {rounds}, train loss: '
            f'{avg_loss:.3f}±{np.std(losses) / avg_loss:.1%}, '
            f'flops: {unit(self.metrics["flops"])}(+{unit(round_flops)}), '
            f'comms: {unit(self.metrics["comms"])}B(+{unit(comms)}B)')
        return info

    def train(self):
        if self.equal_epochs:
            per_epoch = self.client_weights
        else:
            per_epoch = sum(self.client_weights) / self.num_clients
            per_epoch = [per_epoch] * self.num_clients
        steps = [
            math.ceil(p * self.epochs_per_round / self.batch_size)
            for p in per_epoch]
        training = False
        max_rounds = self.max_rounds
        clients, next_clients = None, None
        try:
            clients = self._dispense_clients(clients)
            while True:
                begin_time = time.time()
                # eval
                self.eval(save=training)
                training = True
                self.rounds += 1
                if max_rounds is not None and self.rounds > max_rounds:
                    log.info(
                        f'Maximum number of rounds ({max_rounds}) reached.')
                    break
                # train
                print('debug resampling rounds', self.resampling_rounds)
                if int(self.rounds) % self.resampling_rounds == 0 or next_clients == None:
                    next_clients = self._dispense_clients(count=None, clients=clients, strategy='uniform')
                info = self._round(self.rounds, clients, next_clients, steps)
                clients = next_clients
                log.info(f'{info}, elapsed: {time.time() - begin_time:.2f}s.')
        except KeyboardInterrupt:
            log.info('Abort.')
        return self.finalize()

    def _average(self, states, errors=None, server_state=None, weights=None):
        avg_state = {}
        keys = list(states[list(states)[0]])
        # FIXME this is not correct, as accel uses diff_states to average
        # if errors:
        #     states.update(self.states[e] for e in errors)
        weights = weights or normalize(
            {c: self.client_weights[c] for c in states})
        for k in keys:
            s = [s[k].to(self.device) * weights[c] for c, s in states.items()]
            avg_state[k] = sum(s).to(self.state_device)
            # moving average
            if self.action == 'scaffold':
                server_state[k] = (1-self.server_lr) * server_state[k] + self.server_lr * avg_state[k]
        return server_state if self.action == 'scaffold' else avg_state


    def _duplicate(self, avg_state, next_clients):
        update_states = {}
        for c in next_clients:
            update_states[c] = {
                k: v.detach().clone() for k, v in avg_state.items()}
        return update_states

    def aggregate(self, server_state, states, errors, next_clients):
        avg_state = self._average(states, errors, server_state)
        update_states = self._duplicate(avg_state, next_clients)
        return avg_state, update_states

    def flops(self, results):
        total_flops = sum(r['flops.total'] for r in results.values())
        sample_flops = sum(r['flops.sample'] for r in results.values())
        clients = {str(c): r['flops.model'] for c, r in results.items()}
        flops = {
            'step.clients': clients,
            'step.average': sum(clients.values()) / len(results),
            'total': total_flops,
            'flops.sample':sample_flops,
        }
        return flops



    def communication_cost(self, clients, next_clients, states):
        clients_comm = 0
        self.prune_threshold = 0.65
        self.prune_rate = 0.2 # prune rate for conv layers
        if self.action == 'accel' and self.accel_mode in ['caldas', 'sparse_upload']:
            for c, state in states.items():
                for k, v in state.items():
                    if 'features' in k and 'probs' in k:
                        if self.accel_mode == 'caldas':
                            p = v
                        elif self.accel_mode == 'sparse_upload':
                            p = copy.deepcopy(v)
                            i = int(len(p)*(1-self.prune_rate))
                            ps = sorted(p)
                            ps.reverse()
                            thld = ps[i]
                            p[p<thld] = 0.
                            p[p>=thld] = 1.
                        ratio = p.sum() / len(p)
                        weight = state[k.replace('probs', 'droplayer.weight')]
                        bias = state[k.replace('probs', 'droplayer.bias')]
                        clients_comm += ratio * (weight.nelement()+bias.nelement())
                        if self.accel_mode == 'sparse_upload':
                            rand_state = state[k.replace('probs', 'rand_state')]
                            clients_comm += rand_state.nelement()
                            weight = weight * p.view(weight.shape[0], 1, 1, 1)
                    if 'classifier' in k:
                        clients_comm += state[k].nelement()
            # print(clients_comm)
            if self.accel_mode == 'caldas':
                return clients_comm + len(next_clients) * (clients_comm / len(clients))
            if self.accel_mode == 'sparse_upload':
                per_client_dense = sum(v.nelement() for v in self.server_state.values())
                return clients_comm + len(next_clients) * per_client_dense
        else:
            per_client = sum(v.nelement() for v in self.server_state.values())
            return len(clients) * per_client + len(next_clients) * per_client
