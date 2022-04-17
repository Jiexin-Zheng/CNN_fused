from abc import ABC, abstractmethod

import numpy as np

from initializers import (
    WeightInitializer,
    Stochastic_Gradient_Descent,
    ReLUInitializer,
)

from utils import (
    conv2D,
    im2col,
    col2im,
    pad2D
)


class LayerBase(ABC):
    def __init__(self):
        """An abstract base class inherited by all neural network layers"""
        self.X = []
        self.act_fn = None
        self.trainable = True
        self.optimizer = Stochastic_Gradient_Descent()()

        self.gradients = {}
        self.parameters = {}
        self.derived_variables = {}

        super().__init__()

    @abstractmethod
    def _init_params(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def forward(self, z, **kwargs):
        """Perform a forward pass through the layer"""
        raise NotImplementedError

    @abstractmethod
    def backward(self, out, **kwargs):
        """Perform a backward pass through the layer"""
        raise NotImplementedError

    def freeze(self):
        """
        Freeze the layer parameters at their current values so they can no
        longer be updated.
        """
        self.trainable = False

    def unfreeze(self):
        """Unfreeze the layer parameters so they can be updated."""
        self.trainable = True

    def flush_gradients(self):
        """Erase all the layer's derived variables and gradients."""
        assert self.trainable, "Layer is frozen"
        self.X = []
        for k, v in self.derived_variables.items():
            self.derived_variables[k] = []

        for k, v in self.gradients.items():
            self.gradients[k] = np.zeros_like(v)

    def update(self, cur_loss=None):
        """
        Update the layer parameters using the accrued gradients and layer
        optimizer. Flush all gradients once the update is complete.
        """
        assert self.trainable, "Layer is frozen"
        self.optimizer.step()
        for k, v in self.gradients.items():
            if k in self.parameters:
                self.parameters[k] = self.optimizer(self.parameters[k], v, k, cur_loss)
        self.flush_gradients()

    def set_params(self, summary_dict):
        layer, sd = self, summary_dict

        flatten_keys = ["parameters", "hyperparameters"]
        for k in flatten_keys:
            if k in sd:
                entry = sd[k]
                sd.update(entry)
                del sd[k]

        for k, v in sd.items():
            if k in self.parameters:
                layer.parameters[k] = v
            if k in self.hyperparameters:
                if k == "act_fn":
                    layer.act_fn = ReLUInitializer()()
                elif k == "optimizer":
                    layer.optimizer = Stochastic_Gradient_Descent()()
                elif k not in ["optimizer"]:
                    setattr(layer, k, v)
        return layer

    def summary(self):
        """Return a dict of the layer parameters, hyperparameters, and ID."""
        return {
            "layer": self.hyperparameters["layer"],
            "parameters": self.parameters,
            "hyperparameters": self.hyperparameters,
        }

class BatchNorm2D(LayerBase):
    def __init__(self, momentum=0.9, epsilon=1e-5, optimizer=None):
        super().__init__()

        self.in_ch = None
        self.out_ch = None
        self.epsilon = epsilon
        self.momentum = momentum
        self.parameters = {
            "scaler": None,
            "intercept": None,
            "running_var": None,
            "running_mean": None,
        }
        self.is_initialized = False

    def _init_params(self):
        scaler = np.random.rand(self.in_ch)
        intercept = np.zeros(self.in_ch)

        running_mean = np.zeros(self.in_ch)
        running_var = np.ones(self.in_ch)

        self.parameters = {
            "scaler": scaler,
            "intercept": intercept,
            "running_var": running_var,
            "running_mean": running_mean,
        }

        self.gradients = {
            "scaler": np.zeros_like(scaler),
            "intercept": np.zeros_like(intercept),
        }

        self.is_initialized = True

    @property
    def hyperparameters(self):
        return {
            "layer": "BatchNorm2D",
            "act_fn": None,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "epsilon": self.epsilon,
            "momentum": self.momentum,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def reset_running_stats(self):
        """Reset the running mean and variance estimates to 0 and 1."""
        assert self.trainable, "Layer is frozen"
        self.parameters["running_mean"] = np.zeros(self.in_ch)
        self.parameters["running_var"] = np.ones(self.in_ch)

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.in_ch = self.out_ch = X.shape[3]
            self._init_params()

        ep = self.hyperparameters["epsilon"]
        mm = self.hyperparameters["momentum"]
        rm = self.parameters["running_mean"]
        rv = self.parameters["running_var"]

        scaler = self.parameters["scaler"]
        intercept = self.parameters["intercept"]

        X_mean = self.parameters["running_mean"]
        X_var = self.parameters["running_var"]

        if self.trainable and retain_derived:
            X_mean, X_var = X.mean(axis=(0, 1, 2)), X.var(axis=(0, 1, 2))
            self.parameters["running_mean"] = mm * rm + (1.0 - mm) * X_mean
            self.parameters["running_var"] = mm * rv + (1.0 - mm) * X_var

        if retain_derived:
            self.X.append(X)

        N = (X - X_mean) / np.sqrt(X_var + ep)
        y = scaler * N + intercept
        return y

    def backward(self, dLdy, retain_grads=True):
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        for dy, x in zip(dLdy, X):
            dx, dScaler, dIntercept = self._bwd(dy, x)
            dX.append(dx)

            if retain_grads:
                self.gradients["scaler"] += dScaler
                self.gradients["intercept"] += dIntercept

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X):  # batchnorm -- taking the derivative
        scaler = self.parameters["scaler"]
        ep = self.hyperparameters["epsilon"]

        X_shape = X.shape
        X = np.reshape(X, (-1, X.shape[3]))
        dLdy = np.reshape(dLdy, (-1, dLdy.shape[3]))

        n_ex, in_ch = X.shape
        X_mean, X_var = X.mean(axis=0), X.var(axis=0)

        N = (X - X_mean) / np.sqrt(X_var + ep)
        dIntercept = dLdy.sum(axis=0)
        dScaler = np.sum(dLdy * N, axis=0)

        dN = dLdy * scaler
        dX = (n_ex * dN - dN.sum(axis=0) - N * (dN * N).sum(axis=0)) / (
            n_ex * np.sqrt(X_var + ep)
        )

        return np.reshape(dX, X_shape), dScaler, dIntercept

class Conv2D(LayerBase):
    def __init__(
        self,
        out_ch,
        kernel_shape,
        pad=0,
        stride=1,
        dilation=0,
        act_fn=None,
        optimizer=None,
        init="he_uniform",
    ):
        super().__init__()

        self.pad = pad
        self.init = init
        self.in_ch = None
        self.out_ch = out_ch
        self.stride = stride
        self.dilation = dilation
        self.kernel_shape = kernel_shape
        self.act_fn = ReLUInitializer()()
        self.parameters = {"W": None, "b": None}
        self.is_initialized = False
        self.batchnorm = BatchNorm2D()

    def _init_params(self):
        init_weights = WeightInitializer()  # initialize weights using he_uniform

        fr, fc = self.kernel_shape
        W = init_weights((fr, fc, self.in_ch, self.out_ch))
        b = np.zeros((1, 1, 1, self.out_ch))

        self.parameters = {"W": W, "b": b}
        self.gradients = {"W": np.zeros_like(W), "b": np.zeros_like(b)}
        self.derived_variables = {"Z": [], "out_rows": [], "out_cols": []}
        self.is_initialized = True

    @property
    def hyperparameters(self):
        return {
            "layer": "Conv2D",
            "pad": self.pad,
            "init": self.init,
            "in_ch": self.in_ch,
            "out_ch": self.out_ch,
            "stride": self.stride,
            "dilation": self.dilation,
            "act_fn": str(self.act_fn),
            "kernel_shape": self.kernel_shape,
            "optimizer": {
                "cache": self.optimizer.cache,
                "hyperparameters": self.optimizer.hyperparameters,
            },
        }

    def forward(self, X, retain_derived=True):
        if not self.is_initialized:
            self.in_ch = X.shape[3]
            self._init_params()

        W = self.parameters["W"]
        b = self.parameters["b"]

        n_ex, in_rows, in_cols, in_ch = X.shape
        s, p, d = self.stride, self.pad, self.dilation

        Z = conv2D(X, W, s, p, d) + b
        Z = self.batchnorm.forward(Z)
        Y = self.act_fn(Z)

        if retain_derived:
            self.X.append(X)
            self.derived_variables["Z"].append(Z)
            self.derived_variables["out_rows"].append(Z.shape[1])
            self.derived_variables["out_cols"].append(Z.shape[2])

        return Y

    def backward(self, dLdy, retain_grads=True):
        assert self.trainable, "Layer is frozen"
        if not isinstance(dLdy, list):
            dLdy = [dLdy]

        dX = []
        X = self.X
        Z = self.derived_variables["Z"]

        for dy, x, z in zip(dLdy, X, Z):
            dx, dw, db = self._bwd(dy, x, z)
            dX.append(dx)

            if retain_grads:
                self.gradients["W"] += dw
                self.gradients["b"] += db

        return dX[0] if len(X) == 1 else dX

    def _bwd(self, dLdy, X, Z):
        W = self.parameters["W"]

        d = self.dilation
        fr, fc, in_ch, out_ch = W.shape
        n_ex, out_rows, out_cols, out_ch = dLdy.shape
        (fr, fc), s, p = self.kernel_shape, self.stride, self.pad

        dLdZ = dLdy * self.act_fn.grad(Z)
        dLdZ = dLdZ * self.batchnorm.backward(dLdZ)
        dLdZ_col = dLdZ.transpose(3, 1, 2, 0).reshape(out_ch, -1)
        W_col = W.transpose(3, 2, 0, 1).reshape(out_ch, -1).T
        X_col, p = im2col(X, W.shape, p, s, d)

        dB = dLdZ_col.sum(axis=1).reshape(1, 1, 1, -1)
        dW = (dLdZ_col @ X_col.T).reshape(out_ch, in_ch, fr, fc).transpose(2, 3, 1, 0)

        dX_col = W_col @ dLdZ_col
        dX = col2im(dX_col, X.shape, W.shape, p, s, d).transpose(0, 2, 3, 1)

        return dX, dW, dB
