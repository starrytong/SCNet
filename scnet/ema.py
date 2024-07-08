#From HT demucs https://github.com/facebookresearch/demucs/tree/release_v4?tab=readme-ov-file

from contextlib import contextmanager
import torch
from .utils import swap_state


class ModelEMA:
    """
    Perform EMA on a model. You can switch to the EMA weights temporarily
    with the `swap` method.

        ema = ModelEMA(model)
        with ema.swap():
            # compute valid metrics with averaged model.
    """
    def __init__(self, model, decay=0.9999, unbias=True, device='cpu'):
        self.decay = decay
        self.model = model
        self.state = {}
        self.count = 0
        self.device = device
        self.unbias = unbias

        self._init()

    def _init(self):
        for key, val in self.model.state_dict().items():
            if val.dtype != torch.float32:
                continue
            device = self.device or val.device
            if key not in self.state:
                self.state[key] = val.detach().to(device, copy=True)

    def update(self):
        if self.unbias:
            self.count = self.count * self.decay + 1
            w = 1 / self.count
        else:
            w = 1 - self.decay
        for key, val in self.model.state_dict().items():
            if val.dtype != torch.float32:
                continue
            device = self.device or val.device
            self.state[key].mul_(1 - w)
            self.state[key].add_(val.detach().to(device), alpha=w)

    @contextmanager
    def swap(self):
        with swap_state(self.model, self.state):
            yield

    def state_dict(self):
        return {'state': self.state, 'count': self.count}

    def load_state_dict(self, state):
        self.count = state['count']
        for k, v in state['state'].items():
            self.state[k].copy_(v)
