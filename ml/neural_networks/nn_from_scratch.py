import numpy as np



class NeuralNetworkConfig:
    def __init__(self, layer_sizes : np.ndarray, weights : list[np.ndarray], biases : list[np.ndarray]):
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.biases = biases

    def is_valid(self):
        if not isinstance(self.layer_sizes, np.ndarray):
            return False

        if self.layer_sizes.ndim != 1:
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
    def f(x):
        return 1 / (1 + np.exp(-x))
    
    def __init__(self, config: NeuralNetworkConfig, train_data):
        if not isinstance(config, NeuralNetworkConfig):
            raise TypeError("config must be NeuralNetworkConfig")

        if not config.is_valid():
            raise ValueError("Invalid config")

        self.config = config
        self.train_data = train_data

        self.weights = config.weights
        self.biases = config.biases
        self.N = len(self.weights)

    



layer_sizes = np.array([784, 32, 10])

weights = [
    np.random.randn(32, 784).astype(np.float32),
    np.random.randn(10, 32).astype(np.float32)
]

biases = [
    np.zeros((32, 1), dtype=np.float32),
    np.zeros((10, 1), dtype=np.float32)
]

# train_images = np.loadtxt('data/train_images.txt')
# train_labels = np.loadtxt('data/train_labels.txt').astype(int)
# test_images  = np.loadtxt('data/test_images.txt')
# test_labels  = np.loadtxt('data/test_labels.txt').astype(int)

nn = NeuralNetworkConfig(layer_sizes,weights,biases)
a = NeuralNetwork(nn, None)
