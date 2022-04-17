from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np
from numpy.linalg import norm


class OptimizerBase(ABC):
    def __init__(self, lr):
        """
        An abstract base class for all Optimizer objects.

        This should never be used directly.
        """

        self.cache = {}
        self.cur_step = 0
        self.hyperparameters = {}

    def __call__(self, param, param_grad, param_name, cur_loss=None):
        return self.update(param, param_grad, param_name, cur_loss)

    def step(self):
        """Increment the optimizer step counter by 1"""
        self.cur_step += 1

    def reset_step(self):
        """Reset the step counter to zero"""
        self.cur_step = 0

    def copy(self):
        """Return a copy of the optimizer object"""
        return deepcopy(self)

    def set_params(self, hparam_dict=None, cache_dict=None):
        """Set the parameters of the optimizer object from a dictionary"""

        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v

        if cache_dict is not None:
            for k, v in cache_dict.items():
                if k in self.cache:
                    self.cache[k] = v

    @abstractmethod
    def update(self, param, param_grad, param_name, cur_loss=None):
        raise NotImplementedError


class SGD(OptimizerBase):
    def __init__(
        self, lr=0.01, momentum=0.0, clip_norm=None, **kwargs
    ):
        super().__init__(lr)

        self.hyperparameters = {
            "id": "SGD",
            "lr": lr,
            "momentum": momentum,
            "clip_norm": clip_norm
        }

    def __str__(self):
        H = self.hyperparameters
        lr, mm, cn = H["lr"], H["momentum"], H["clip_norm"]
        return "SGD(lr={}, momentum={}, clip_norm={})".format(
            lr, mm, cn
        )

    def update(self, param, param_grad, param_name, cur_loss=None):

        C = self.cache
        H = self.hyperparameters
        momentum, clip_norm = H["momentum"], H["clip_norm"]
        lr = H['lr']

        if param_name not in C:
            C[param_name] = np.zeros_like(param_grad)

        # scale gradient to avoid explosion
        t = np.inf if clip_norm is None else clip_norm
        if norm(param_grad) > t:
            param_grad = param_grad * t / norm(param_grad)

        update = momentum * C[param_name] + lr * param_grad
        self.cache[param_name] = update
        return param - update