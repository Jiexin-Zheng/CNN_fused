import numpy as np

from optimizers import SGD
from activations import ReLU

from utils import he_uniform


class ReLUInitializer(object):

    def __call__(self):

        act_fn = ReLU()

        return act_fn


class Stochastic_Gradient_Descent(object):

    def __call__(self):
        """Initialize the optimizer"""
       
        opt = SGD()
        
        return opt


class WeightInitializer(object):

    def __call__(self, weight_shape):

        W = he_uniform(weight_shape)

        return W
