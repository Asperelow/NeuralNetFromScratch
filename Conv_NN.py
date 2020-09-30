import sys
import numpy as np
from numba import jit, cuda, njit, float64, int64, void
import math
import matplotlib.pyplot as plt


def activation_fwd(Z, activation):
    """
    Arguments:
    Z -- Value to be fed to the activation
    activation -- Type of activation to be performed on Z
    Returns:
    A -- The activation performed on Z
    """
    if activation == "sigmoid":
        pos_mask = (Z >= 0)
        neg_mask = (Z < 0)
        z = np.zeros_like(Z)
        z[pos_mask] = np.exp(-Z[pos_mask])
        z[neg_mask] = np.exp(Z[neg_mask])
        top = np.ones_like(Z)
        top[neg_mask] = z[neg_mask]
        return top / (1 + z)
        return 1. / (1. + np.exp(-Z))
    elif activation == "relu":
        return np.maximum(0, Z)


def activation_back(Z, activation):
    """
    Arguments:
    Z -- Value to be fed to the activation derivative
    activation -- Type of activation derivative to be performed on Z
    Returns:
    A -- The activation derivative performed on Z
    """
    if activation == "sigmoid":
        sig = activation_fwd(Z, "sigmoid")
        return sig * (1 - sig)
    elif activation == "relu":
        return (Z > 0).astype(int)


def zero_pad(X, pad):
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad


def init_parameters():
    """
    Arguments:
    layer_dims -- number of 'neurons' in each layer, x and y inclusive
    Returns:
    parameters -- Initialized dictionary of 'W' and 'b' values for each layer
    """
    L_dims = len(layer_dims)  # Finding the number of layers in the network
    parameters = {}
    activations["layer0"] = 'NULL'  # Avoids loop problems
    for l in range(1, L_dims):
        if activations["layer" + str(l)] != 'conv':
            parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))

            if activations["layer" + str(l - 1)] == 'conv':
                layer_dims_prev = x.shape[0] * conv_layer["layer" + str(l - 1)][2]
            else:
                layer_dims_prev = layer_dims[l - 1]
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims_prev) * 0.01
        else:
            parameters["b" + str(l)] = np.zeros((conv_layer["layer1"][2], 1))
            parameters["W" + str(l)] = np.random.randn(layer_dims[l], 1) * 0.01
    return parameters


def mini_batch_maker(X, Y):
    """
    Arguments:
    X -- A list of inputs into the Neural Network
    Y -- The value the network should output
    mini_batch_size -- The size of each mini-batch
    Returns:
    mini_batches -- list of mini batches X and Y
    """
    m = X.shape[1]
    mini_batches = []
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_full_minibatches = math.floor(m / mini_batch_size)
    for b in range(0, num_full_minibatches):
        mini_batch_X = shuffled_X[:, b * mini_batch_size:(b + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, b * mini_batch_size:(b + 1) * mini_batch_size]
        mini_batches.append((mini_batch_X, mini_batch_Y))
        if m % mini_batch_size != 0:
            end = m - mini_batch_size * num_full_minibatches
            mini_batch_X = shuffled_X[:, num_full_minibatches * mini_batch_size]
            mini_batch_Y = shuffled_Y[:, num_full_minibatches * mini_batch_size]
            mini_batches.append((mini_batch_X, mini_batch_Y))
    return mini_batches


def forward_prop(X, parameters):
    """
    Arguments:
    X -- A list of inputs into the Neural Network
    parameters -- Dictionary of 'W' and 'b' values for each layer
    activations -- Type of activation to be performed on Z for each layer
    Returns:
    A_cache -- The activations performed on Z for each layer, used in backprop
    Z_cache -- The Z values for each layer, used in backprop
    """
    A = X  # x is the first Activation layer, so it is set equal to the first A
    cache = {}
    for l in range(1, L + 1):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        if activations["layer" + str(l)] != 'conv':
            Z = linear_forward(A_prev, W, b)
        else:
            if l == 1:
                A_prev = A_prev.T
                A_prev = A_prev.reshape(A_prev.shape[0], 28, 28, 1)
                W = W.reshape(conv_layer["layer1"][0], conv_layer["layer1"][1], 1,
                              conv_layer["layer1"][2])
                b = b.reshape(1, 1, 1, conv_layer["layer1"][2])
                hparameters_conv = hparameters_conv_layers["layer" + str(l)]
            Z = conv_forward(A_prev, W, b, hparameters_conv)
            conv_layer_shapes["layer" + str(l)] = Z.shape
            Z = Z.reshape(int(Z.size / mini_batch_size), mini_batch_size)

        if activations["layer" + str(l)] == "sigmoid":
            A = activation_fwd(Z, 'sigmoid')
            A = A - ((A == 1).astype(int) * 0.001) + ((A == 0).astype(int) * 0.001)

        elif activations["layer" + str(l)] == "relu" or activations["layer" + str(l)] == "conv":
            A = activation_fwd(Z, 'relu')
        cache["A" + str(l)] = A
        cache["Z" + str(l)] = Z
    return cache


def linear_forward(A, W, b):
    Z = np.dot(W, A) + b
    return Z


def conv_forward(A_prev, W, b, hparameters_conv):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters_conv["stride"]
    pad = hparameters_conv["pad"]

    n_H = int((n_H_prev - f + 2 * pad) / stride) + 1
    n_W = int((n_W_prev - f + 2 * pad) / stride) + 1
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start: horiz_end, :]
                    Z[i, h, w, c] = np.sum(np.multiply(a_slice_prev, W) + b)
    return Z


def backprop(X, Y, parameters, cache):
    """
    Arguments:
    Y -- The value the network should output
    parameters -- Dictionary of 'W' and 'b' values for each layer
    A_cache -- The activations performed on Z for each layer, used in backprop
    Z_cache -- The Z values for each layer, used in backprop
    activations -- Type of activation to be performed on Z for each layer
    Returns:
    grads -- dictionary of 'dA','dW', and 'db'. Used for updating the parameters
    """
    grads = {}
    dZ = {}
    dA = {}
    L = len(parameters) // 2
    dZ[str(L)] = abs(Y-cache["A"+str(L)])
    for l in reversed(range(1, L + 1)):  # l= last later --> l=0
        if l == 1:
            if activations["layer" + str(l)] == 'conv':
                hparameters_conv = hparameters_conv_layers["layer" + str(l)]
                stride = hparameters_conv["stride"]
                pad = hparameters_conv["pad"]
                dZ = dZ[str(l)].T
                dZ = dZ.reshape(mini_batch_size,
                                conv_layer_shapes["layer1"][1],
                                conv_layer_shapes["layer1"][2],
                                conv_layer["layer1"][2])
                X = X.T.reshape(conv_layer_shapes["layer0"])
                X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
                dX = np.zeros(X.shape)
                dX_pad = np.pad(dX, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')

                W = parameters["W" + str(l)].reshape(conv_layer["layer" + str(l)][0],
                                                     conv_layer["layer" + str(l)][1], 1,
                                                     conv_layer["layer" + str(l)][2])

                grads["dW" + str(l)], grads["db" + str(l)] = conv_backprop(dZ, X, W, stride, pad,
                                                                                           X_pad, dX, dX_pad)

                grads["dW" + str(l)] = grads["dW" + str(l)].reshape(parameters["W" + str(l)].shape)
                grads["db" + str(l)] = grads["db" + str(l)].reshape(parameters["b" + str(l)].shape)

            else:
                grads["dW" + str(l)], grads["db" + str(l)], dA[str(l - 1)] = linear_backprop(dZ[str(l)], X,
                                                                                             parameters["W" + str(l)])
        else:
            if activations["layer" + str(l)] == 'conv':
                hparameters_conv = hparameters_conv_layers["layer" + str(l)]
                stride = hparameters_conv["stride"]
                pad = hparameters_conv["pad"]
                dZ = dZ[str(l)].T
                dZ = dZ.reshape(mini_batch_size,
                                conv_layer_shapes["layer" + str(l)][1],
                                conv_layer_shapes["layer" + str(l)][2],
                                conv_layer["layer" + str(l)][2])
                A_last = cache["A" + str(l - 1)].T
                A_last = A_last.reshape(conv_layer_shapes["layer" + str(l - 1)])
                W = parameters["W" + str(l)].reshape(conv_layer["layer" + str(l)][0],
                                                     conv_layer["layer" + str(l)][1], 1,
                                                     conv_layer["layer" + str(l)][2])

                grads["dW" + str(l)], grads["db" + str(l)], dA[str(l - 1)] = conv_backprop(dZ[str(l)], A_last, W,
                                                                                           stride, pad)
                # Vectorize gradients dW, db and dA
                grads["dW" + str(l)] = grads["dW" + str(l)].reshape(parameters["W" + str(l)].shape)
                grads["db" + str(l)] = grads["db" + str(l)].reshape(parameters["b" + str(l)].shape)
                dA[str(l - 1)] = dA[str(l - 1)].reshape(A_last.shape)

            else:  # relu or sigmoid
                grads["dW" + str(l)], grads["db" + str(l)], dA[str(l - 1)] = linear_backprop(dZ[str(l)],
                                                                                             cache["A" + str(l - 1)],
                                                                                             parameters["W" + str(l)])
        if 1 < l < L + 1:
            if activations["layer" + str(l - 1)] == 'sigmoid':
                dZ[str(l - 1)] = dA[str(l - 1)] * activation_back(cache["Z" + str(l - 1)], 'sigmoid')
            elif activations["layer" + str(l - 1)] == 'relu' or activations["layer" + str(l - 1)] == 'conv':
                dZ[str(l - 1)] = dA[str(l - 1)] * activation_back(cache["Z" + str(l - 1)], 'relu')
    return grads


def linear_backprop(dZ, A, W):
    m = A.shape[1]
    dW = np.dot(dZ, A.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    return dW, db, dA_prev


@jit()
def conv_backprop(dZ, A_prev, W, stride, pad, A_prev_pad, dA_prev, dA_prev_pad):
    (m, n_H, n_W, n_C) = dZ.shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    dW = np.zeros((f, f, n_C_prev, n_C))
    db = np.sum(dZ, axis=(0, 1, 2))
    db.reshape(n_C, -1)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    dW[:, :, :, c] += a_slice * dZ[i, h, w, c]
    return dW, db


def compute_cost(cache, Y):
    """
    Arguments:
    A_cache -- The activations performed on Z for each layer, used in backprop
    Y -- The value the network should output
    L -- The number of layers in the network
    Returns:
    cost -- The total cost of the network
    """
    A_last = cache["A" + str(L)]
    m = Y.size
    cost = -(1 / m) * np.sum(Y * np.log(A_last) + (1 - Y) * np.log(1 - A_last), axis=1, keepdims=True)
    cost = np.squeeze(cost)
    return np.sum(cost)


def initialize_hyperparameters(parameters):
    """
    Arguments:
    parameters -- Dictionary of 'W' and 'b' values for each layer
    Returns:
    hparameters -- Dictionary of 'V_dW' values for each layer, the momentum value
    """
    hparameters = {}
    for l in range(L):
        hparameters["V_dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        hparameters["V_db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return hparameters


def update_parameters(parameters, grads, hparameters):
    """
    Arguments:
    parameters -- Dictionary of 'W' and 'b' values for each layer
    grads -- Results from backprop. 'dA', 'dW', and 'db' values for each layer
    update_parameter_values -- Learning rate and momentum values
    hparameters -- Dictionary of current momentum values
    Returns:
    parameters -- Dictionary of updated parameter values, 'W' and 'b' for each layer
    """
    L = len(parameters) // 2
    for l in range(L):
        hparameters["V_dW" + str(l + 1)] = beta * hparameters["V_dW" + str(l + 1)] + (1 - beta) * grads[
            "dW" + str(l + 1)]
        hparameters["V_db" + str(l + 1)] = beta * hparameters["V_db" + str(l + 1)] + (1 - beta) * grads[
            "db" + str(l + 1)]

        dropout_W = (np.random.uniform(0, 1, parameters["W" + str(l + 1)].shape[0]) > dropout_rate).astype(int)
        hparameters["V_dW" + str(l + 1)] = (hparameters["V_dW" + str(l + 1)].T * dropout_W).T
        dropout_b = (np.random.uniform(0, 1, parameters["b" + str(l + 1)].shape[0]) > dropout_rate).astype(int)
        hparameters["V_db" + str(l + 1)] = (hparameters["V_db" + str(l + 1)].T * dropout_b).T
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * hparameters["V_dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * hparameters["V_db" + str(l + 1)]
    return parameters


def train_nn(mini_batches):
    """
    Arguments:
    layer_dims -- number of 'neurons' in each layer, x and y inclusive
    X -- A list of inputs for the Neural Network to be trained on
    Y -- The value the network should output
    stopping_value -- The maximum value the cost can change by before stopping
    update_parameter_values -- Learning rate and momentum values
    activations -- Type of activation to be performed on Z for each layer
    Returns:
    parameters -- Dictionary of 'W' and 'b' values for each layer
    """
    parameters = init_parameters()
    hparameters = initialize_hyperparameters(parameters)
    for iters in range(iterations):
        for batch in range(len(mini_batches)):
            X, Y = mini_batches[batch]
            cache = forward_prop(X, parameters)
            cost = compute_cost(cache, Y)

            sys.stdout.write('\r')  # Makes it so there isn't a wall of text on the screen
            sys.stdout.write("Cost = " + str(cost))
            sys.stdout.flush()

            grads = backprop(X, Y, parameters, cache)
            parameters = update_parameters(parameters, grads, hparameters)
    return parameters


samples = 1000  # Number of samples to read in
tt_ratio = 0.8  # train-test ratio
conv_layer_shapes = {}

data = np.loadtxt(open("train.csv", "rb"), delimiter=",", skiprows=1, max_rows=(samples + 1))
x = np.array(data[0:int(samples * tt_ratio), 1:]).T / 255
y_temp = np.array(data[0:int(samples * tt_ratio), 0]).reshape(-1)
y = np.eye(10)[y_temp.astype(int)].T  # Creates a one hot array. 10 classes, y_temp is the target

x_test = np.array(data[int(samples * tt_ratio):samples, 1:]).T / 255
y_test_temp = np.array(data[int(samples * tt_ratio):samples, 0]).reshape(-1)
y_test = np.eye(10)[y_test_temp.astype(int)].T

mini_batch_size = 20
mini_batches_m = mini_batch_maker(x, y)

conv_layer_shapes["layer0"] = list((x.T.reshape(int(samples * tt_ratio), 28, 28, 1)).shape)  # m, n_H, n_W, n_C
conv_layer_shapes["layer0"][0] = int(mini_batch_size)

conv_layer = {"layer1": [3, 3, 64]}  # f, f, n_C
hparameters_conv_layers = {"layer1": {"pad": 1,
                                      "stride": 1}}
layer_dims = [x.shape[0], np.prod(conv_layer["layer1"]), 128, y.shape[0]]
L = len(layer_dims) - 1

activations = {"layer1": "conv",  # Activations for the hidden layers and output
               "layer2": "relu",
               "layer3": "sigmoid"}

iterations = 5000
learning_rate = 1e-4
beta = 0.9
dropout_rate = 0.5

parameters_m = train_nn(mini_batches_m)

guess = forward_prop(x_test, parameters_m)
guess = guess["Z" + str(L)].T
guess = (guess == guess.max(axis=1)[:, None]).astype(int)
compare = guess.T + y_test
score = sum(sum((compare == 2).astype(int))) / (samples * 0.5)
print("\nScore:" + str(score * 100) + "%")


