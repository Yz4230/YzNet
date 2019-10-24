import numpy as np
from abc import abstractmethod, ABCMeta
import matplotlib.pyplot as plt


class NetworkComponent(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dx: np.ndarray) -> np.ndarray:
        pass


class Neuron(NetworkComponent):
    def __init__(self, input_dim: int, output_dim: int, lr: float):
        self.weight = np.random.randn(output_dim, input_dim)
        self.input = np.zeros((input_dim, 1))
        self.grad = np.zeros(self.weight.shape)
        self.lr = lr

    def forward(self, x: np.ndarray):
        if self.input.shape != x.shape:
            raise ValueError(f"Expected shape {self.input.shape}, but input shape {x.shape}")
        self.input = x
        return np.dot(self.weight, self.input)

    def backward(self, dx: np.ndarray):
        self.grad = np.dot(dx, self.input.reshape(1, -1))
        return np.dot(self.weight.T, dx)

    def update(self):
        self.weight -= self.grad * self.lr


class ReLU(NetworkComponent):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, dx: np.ndarray) -> np.ndarray:
        return np.where(dx > 0, 1, 0)


class Sigmoid(NetworkComponent):
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dx: np.ndarray):
        return self.output * (1 - self.output)


class MSE:
    @classmethod
    def forward(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        return np.mean(np.square(y.flatten() - x.flatten())) / 2

    @classmethod
    def backward(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.shape != y.shape:
            raise ValueError(f"Invalid shape, {x.shape} != {y.shape}")
        return (x.reshape(-1, 1) - y.reshape(-1, 1)) / x.size


class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.history = []

    def add_layer(self, layer: NetworkComponent) -> None:
        self.layers.append(layer)

    def learn(self, epoch: int, x: np.ndarray, y: np.ndarray) -> list:
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Invalid x or y, {x.shape[0]} != {y.shape[0]}")
        self.history.clear()
        data_size = x.shape[0]
        for i in range(epoch):
            loss_sum = 0
            for xi, yi in zip(x, y):
                feed_forward = self.layers[0].forward(xi)
                for l in self.layers[1:]:
                    feed_forward = l.forward(feed_forward)
                loss = MSE.forward(feed_forward, yi)
                loss_sum += loss
                grad = MSE.backward(feed_forward, yi)
                grad = self.layers[-1].backward(grad)
                for l in reversed(self.layers[:-1]):
                    grad = l.backward(grad)
                for l in self.layers:
                    if isinstance(l, Neuron):
                        l.update()
            self.history.append(loss_sum / data_size)
            print(f"Loss = {self.history[-1]}")
        return self.history

    def plot_loss(self):
        plt.plot(self.history)
        plt.show()

    def load_model(self):
        pass

    def load_weight(self):
        pass
