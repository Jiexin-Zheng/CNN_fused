from abc import ABC, abstractmethod

import numpy as np


class ActivationBase(ABC):
    def __init__(self, **kwargs):
        """Initialize the ActivationBase object"""
        super().__init__()

    def __call__(self, z):
        """Apply the activation function to an input"""
        if z.ndim == 1:
            z = z.reshape(1, -1)
        return self.fn(z)

    @abstractmethod
    def fn(self, z):
        """Apply the activation function to an input"""
        raise NotImplementedError

    @abstractmethod
    def grad(self, x, **kwargs):
        """Compute the gradient of the activation function wrt the input"""
        raise NotImplementedError

class ReLU(ActivationBase):
    def __init__(self):
        super().__init__()

    def __str__(self):
        """Return a string representation of the activation function"""
        return "ReLU"

    def fn(self, z):
        return np.clip(z, 0, np.inf)

    def grad(self, x):
        return (x > 0).astype(int)

    def grad2(self, x):
        return np.zeros_like(x)