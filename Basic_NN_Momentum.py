import sys
import numpy as np
import math


x = np.random.randint(-100, 100, [1, 10000]) * np.exp(np.random.randint(-4, 1, [1, 10000]))
x_temp = np.sum(x, axis=0, keepdims=True)

y_temp = (x_temp > 0).astype(int)
y = np.eye(2)[y_temp.astype(int)].T  # Creates a one hot array. 10 classes, y_temp is the target
y = np.squeeze(y)
################################ Adjustable Parameters ################################
layer_dims_m = [x.shape[0], 20, y.shape[0]] # First and last entry should be x and y respectively
activations_m = {"layer1": "relu",          # Activations for the hidden layers and output
                 "layer2": "sigmoid"}
iteration_m = 50000                         # The number of iterations this program will train
learning_rate_m = 0.001                     # How quickly the program will learn (too high and it will become unbounded)
beta_m = 0.9                                # Momentum parameter, 0.9 is a good value
mini_batche_size_m = 1000                   # Size of each mini-batch
#######################################################################################


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


def init_parameters(layer_dims):
    """
    Arguments:
    layer_dims -- number of 'neurons' in each layer, x and y inclusive
    Returns:
    parameters -- Initialized dictionary of 'W' and 'b' values for each layer
    """
    L_dims = len(layer_dims)    # Finding the number of layers in the network
    parameters = {}             # Initializing a parameters dictionary
    for l in range(1, L_dims):
        parameters["W" + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters


def mini_batch_maker(X, Y, mini_batch_size):
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


def forward_prop(X, parameters, activations):
    """
    Arguments:
    X -- A list of inputs into the Neural Network
    parameters -- Dictionary of 'W' and 'b' values for each layer
    activations -- Type of activation to be performed on Z for each layer
    Returns:
    A_cache -- The activations performed on Z for each layer, used in backprop
    Z_cache -- The Z values for each layer, used in backprop
    """
    A = X                       # x is the first Activation layer, so it is set equal to the first A
    L = len(parameters) // 2    # Finding the number of layers in the network
    cache = {}
    for l in range(1, L + 1):
        A_prev = A
        W = parameters["W" + str(l)]
        b = parameters["b" + str(l)]
        Z = np.dot(W, A_prev) + b
        if activations["layer" + str(l)] == "sigmoid":
            A = activation_fwd(Z, activations["layer" + str(l)])
            A = A - ((A == 1).astype(int) * 0.001)  # Prevents log(0) which is NaN
            A = A + ((A == 0).astype(int) * 0.001)  # Prevents log(0) which is NaN
        elif activations["layer" + str(l)] == "relu":
            A = activation_fwd(Z, activations["layer" + str(l)])
        cache["A" + str(l)] = A
        cache["Z" + str(l)] = Z
    return cache


def backprop(X, Y, parameters, cache, activations):
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
    L = len(parameters) // 2
    m = Y.shape[1]
    dZ[str(L)] = (cache["A" + str(L)] - Y) * activation_back(cache["Z"+str(L)],'sigmoid')

    for l in reversed(range(1, L+1)):  # l= last later --> l=0
        if l == 1:
            grads["dW" + str(l)] = (1 / m) * np.dot(dZ[str(l)], X.T)
        else:
            grads["dW" + str(l)] = (1 / m) * np.dot(dZ[str(l)], cache["A" + str(l-1)].T)
        grads["db" + str(l)] = (1 / m) * np.sum(dZ[str(l)], axis=1, keepdims=True)
        if 1 < l < L + 1:
            if activations["layer" + str(l-1)] == "sigmoid":
                dZ[str(l-1)] = np.dot(parameters["W" + str(l)].T, dZ[str(l)]) * activation_back(
                    cache["Z" + str(l-1)], "sigmoid")
            elif activations["layer" + str(l-1)] == "relu":
                dZ[str(l-1)] = np.dot(parameters["W" + str(l)].T, dZ[str(l)]) * activation_back(
                    cache["Z" + str(l-1)], "relu")
    return grads


def compute_cost(cache, Y, L):
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
    L = len(parameters) // 2
    hparameters = {}
    for l in range(L):
        hparameters["V_dW" + str(l + 1)] = np.zeros_like(parameters["W" + str(l + 1)])
        hparameters["V_db" + str(l + 1)] = np.zeros_like(parameters["b" + str(l + 1)])
    return hparameters


def update_parameters(parameters, grads, update_parameter_values, hparameters):
    """
    Arguments:
    parameters -- Dictionary of 'W' and 'b' values for each layer
    grads -- Results from backprop. 'dA', 'dW', and 'db' values for each layer
    update_parameter_values -- Learning rate and momentum values
    hparameters -- Dictionary of current momentum values
    Returns:
    parameters -- Dictionary of updated parameter values, 'W' and 'b' for each layer
    """
    learning_rate, beta = update_parameter_values
    L = len(parameters) // 2
    for l in range(L):
        hparameters["V_dW" + str(l + 1)] = beta * hparameters["V_dW" + str(l + 1)] + (1 - beta) * grads[
            "dW" + str(l + 1)]
        hparameters["V_db" + str(l + 1)] = beta * hparameters["V_db" + str(l + 1)] + (1 - beta) * grads[
            "db" + str(l + 1)]
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * hparameters["V_dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * hparameters["V_db" + str(l + 1)]
    return parameters


def train_nn(layer_dims, mini_batches, iteration, update_parameter_values, activations):
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
    parameters = init_parameters(layer_dims)
    L = len(parameters) // 2
    hparameters = initialize_hyperparameters(parameters)
    for iter in range(iteration):
        for batch in range(0, len(x)):
            X, Y = mini_batches[batch]
            cache = forward_prop(X, parameters, activations)
            cost = compute_cost(cache, Y, L)
            sys.stdout.write('\r')  # Makes it so there isn't a wall of text on the screen
            sys.stdout.write("Cost = " + str(cost))
            sys.stdout.flush()

            grads = backprop(X, Y, parameters, cache, activations)
            parameters = update_parameters(parameters, grads, update_parameter_values, hparameters)
    return parameters


mini_batches_m = mini_batch_maker(x, y, mini_batche_size_m)
update_parameter_values_m = learning_rate_m, beta_m
parameters_m = train_nn(layer_dims_m, mini_batches_m, iteration_m, update_parameter_values_m, activations_m)

while True:
    guess = float(input("\nEnter a guess:"))
    cache_m = forward_prop(guess, parameters_m, activations_m)
    guess = cache_m["A" + str(len(layer_dims_m) - 1)]
    if guess[1] > guess[0]:
        print("positive")
    else:
        print("negative")
