from layers import Conv2D
from activations import ReLU
import numpy as np


def random_tensor(shape): # generating random vectors
    offset = np.random.randint(-300, 300, shape)
    X = np.random.rand(*shape) + offset
    eps = np.finfo(float).eps
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + eps)
    return X


def fuse(conv, bn, n_out, f_shape, act_fn, p, s, d):
    w = conv.parameters["W"]
    mean = bn.parameters["running_mean"]
    var_sqrt = np.sqrt(bn.parameters["running_var"])

    beta = bn.parameters["scaler"]
    gamma = bn.parameters["intercept"]

    b = conv.parameters["b"]

    w = w * (beta / var_sqrt)
    b = (b - mean)/var_sqrt * beta + gamma

    fused_conv = Conv2D(
            out_ch=n_out,
            kernel_shape=f_shape,
            act_fn=act_fn,
            pad=p,
            stride=s,
            dilation=d,
        )
    fused_conv.parameters["W"] = w
    fused_conv.parameters["b"] = b
    return fused_conv


if __name__ == "__main__":
    N = 15  # testing time

    np.random.seed(12345)  # random seed fixed

    i = 1
    while i < N + 1:
        n_ex = np.random.randint(1, 10)  # batch_size
        in_rows = np.random.randint(1, 224) # row size
        in_cols = np.random.randint(1, 224) # column size
        n_in, n_out = np.random.randint(1, 3), np.random.randint(1, 3)  # input and output channel size
        f_shape = (  # kernel size
            min(in_rows, np.random.randint(1, 5)),
            min(in_cols, np.random.randint(1, 5)),
        )
        p, s = np.random.randint(0, 5), np.random.randint(1, 3) # padding and stride
        d = np.random.randint(0, 5) # dilation rate

        fr, fc = f_shape[0] * (d + 1) - d, f_shape[1] * (d + 1) - d # Dilated Convolution
        out_rows = int(1 + (in_rows + 2 * p - fr) / s)  # row size after convolution
        out_cols = int(1 + (in_cols + 2 * p - fc) / s)  # column size after convolution

        if out_rows <= 0 or out_cols <= 0:
            continue

        X = random_tensor((n_ex, in_rows, in_cols, n_in))  # input matrix

        # ReLU activation function
        act_fn = ReLU()

        L1 = Conv2D(
            out_ch=n_out,
            kernel_shape=f_shape,
            act_fn=act_fn,
            pad=p,
            stride=s,
            dilation=d,
        )
        # forward prop
        import time
        start = time.time()
        y_pred = L1.forward(X)
        end = time.time()
      
        fusion = fuse(L1, L1.batchnorm, n_out, f_shape, act_fn, p, s, d)
        start_fuse = time.time()
        y_pred_fused = fusion.forward(X)
        end_fuse = time.time()
        
            # backprop
        dLdy = np.ones_like(y_pred)
        dLdX = L1.backward(dLdy)

        params = [
            (L1.X[0], "X"),
            (y_pred, "y"),
            (y_pred_fused, "y_fused"),
            (L1.parameters["W"], "W"),
            (L1.parameters["b"], "b"),
            (L1.gradients["W"], "dLdW"),
            (L1.gradients["b"], "dLdB"),
            (dLdX, "dLdX"),
        ]

        print("\nTrial {}".format(i))
        print("Original time: ", (end - start)*1000)
        print("Fused time: ", (end_fuse - start_fuse)*1000)
        for ix, (mine, label) in enumerate(params):
            print("\t {}: {}".format(label, mine.shape))
        i += 1