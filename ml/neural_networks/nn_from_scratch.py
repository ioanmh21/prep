import numpy as np

class NeuralNetworkConfig:
    def __init__(self, layer_sizes : np.ndarray, weights : list[np.ndarray] = None, biases : list[np.ndarray] = None):
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.biases = biases

    def is_valid(self):
        if not isinstance(self.layer_sizes, np.ndarray):
            return False

        if self.layer_sizes.ndim != 1:
            return False
        
        if self.layer_sizes.size < 2:
            return False
        
        if np.any(self.layer_sizes <= 0):
            return False
        
        N = self.layer_sizes.size
        if len(self.weights) != N - 1 or len(self.biases) != N - 1:
            return False
        
        for i in range(1, N):
            if not isinstance(self.weights[i - 1], np.ndarray):
                return False
            if not isinstance(self.biases[i - 1], np.ndarray):
                return False

            if self.weights[i - 1].ndim != 2 or self.biases[i - 1].ndim != 2:
                return False

            mat_shape = self.weights[i - 1].shape

            if mat_shape[1] != self.layer_sizes[i - 1] or mat_shape[0] != self.layer_sizes[i]:
                return False
            
            if self.biases[i - 1].shape != (mat_shape[0], 1):
                return False

        return True


class NeuralNetwork:
    def __init__(self, config: NeuralNetworkConfig, train_data, train_labels):
        if not isinstance(config, NeuralNetworkConfig):
            raise TypeError("config must be NeuralNetworkConfig")

        if not config.is_valid():
            raise ValueError("Invalid config")

        self.config = config
        self.train_data = train_data
        self.train_labels = self.one_hot_encoding(train_labels)

        self.N = config.layer_sizes.size - 1

        if config.weights is None:
            self.weights = []

            for i in range(1, config.layer_sizes.size):
                rows = config.layer_sizes[i]
                cols = config.layer_sizes[i - 1]

                W = (
                    np.random.randn(rows, cols)
                    * 0.01
                ).astype(np.float32)
                self.weights.append(W)
        else:
            self.weights = config.weights

        if config.biases is None:
            self.biases = []

            for i in range(1, config.layer_sizes.size):
                rows = config.layer_sizes[i]

                b = np.zeros(
                    (rows, 1),
                    dtype = np.float32
                )
                self.biases.append(b)

        else:
            self.biases = config.biases

        self.config.weights = self.weights
        self.config.biases = self.biases

        if not self.config.is_valid():
            raise ValueError("Invalid config")


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def relu(self, x):
        return np.maximum(0, x) 
    def softmax(self, x):
        x = x - np.max(x, axis = 0, keepdims = True)
        exps = np.exp(x)
        return exps / np.sum(exps, axis = 0, keepdims = True)
    
    def mse(self, x_real, x_ideal):
        if x_real.shape != x_ideal.shape:
            raise ValueError("Shape mismatch")

        return np.mean(
            (x_real - x_ideal) ** 2
        )

    def one_hot_encoding(self, train_labels):
        m = train_labels.size
        Y = np.zeros((self.config.layer_sizes[-1], m), dtype = np.float32)
        Y[train_labels, np.arange(m)] = 1
        return Y

    def forward(self, input):
        if not isinstance(input, np.ndarray):
            raise ValueError("input must be numpt ndarray")
        if input.shape[0] != self.weights[0].shape[1]:
            raise ValueError("Invalid input shape")

        activations = [input]
        zs = []

        A = input
        for i in range(self.N - 1):
            Z = self.weights[i] @ A + self.biases[i]
            A = self.sigmoid(Z)

            activations.append(A)  
            zs.append(Z)

        Z = self.weights[-1] @ A + self.biases[-1]
        A = self.softmax(Z)

        activations.append(A)
        zs.append(Z)

    def train(self, epoch_number = 1, batch_size = 32):
        m = self.train_data.shape[1]
        for _ in range(epoch_number):
            perm = np.random.permutation(len(self.train_data))

            X = self.train_data[:, perm]
            Y = self.train_labels[:, perm]

            for start in range(0, m, batch_size):
                end = start + batch_size

                X_batch = X[:, start:end]
                Y_batch = Y[:, start:end]

                activations, zs = self.forward(X_batch)
                loss = self.mse(activations[-1], Y_batch)


# layer_sizes = np.array([784, 32, 10])

# weights = [
#     np.random.randn(32, 784).astype(np.float32),
#     np.random.randn(10, 32).astype(np.float32)
# ]

# biases = [
#     np.zeros((32, 1), dtype=np.float32),
#     np.zeros((10, 1), dtype=np.float32)
# ]

# train_images = np.loadtxt('data/train_images.txt').T
# train_labels = np.loadtxt('data/train_labels.txt').T.astype(np.int32)
# test_images  = np.loadtxt('data/test_images.txt')
# test_labels  = np.loadtxt('data/test_labels.txt').astype(np.int32)

# nn = NeuralNetworkConfig(layer_sizes,weights,biases)
# a = NeuralNetwork(nn, train_images, train_labels)
# print(a.train_data.shape)