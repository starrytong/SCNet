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

# Some or all of the work in this file may be restricted by the following copyright.
"""
MIT License

Copyright (c) Meta, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
