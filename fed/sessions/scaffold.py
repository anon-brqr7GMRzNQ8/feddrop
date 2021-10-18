import torch
from .sync import Synchronous
from ..utils import normalize



class Scaffold(Synchronous):
    def __init__(
            self, *args, server_lr=0.1,
            **kwargs):
        self.server_lr = server_lr
        super().__init__(*args, **kwargs)
        self.client_cvs = {}
        self.server_cv = []

    def aggregate(self, server_state, states, errors, next_clients):
        avg_state = self._average(states, errors, server_state)
        update_states = self._duplicate(avg_state, next_clients)
        self.update_server_cv()
        return avg_state, update_states

    def update_server_cv(self,):
        avg_delta_cv = []
        cvs = [cv for cv in self.delta_cv.values()] # dict -> list
        for i in range(len(cvs[0])):
            avg_delta_cv.append( torch.sum(torch.cat([cv[i] for cv in cvs], dim=0), dim=0) / self.num_clients )
        self.server_cv += avg_delta_cv
