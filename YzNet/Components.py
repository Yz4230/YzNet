from abc import abstractmethod, ABCMeta
import numpy as np


class NetworkComponent(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, dx: np.ndarray) -> np.ndarray:
        pass

    @classmethod
    def get_layer_name(cls) -> str:
        return cls.__name__

    def __str__(self):
        return self.get_layer_name()


class Neuron(NetworkComponent):
    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = np.random.randn(output_dim, input_dim)
        self.input = np.zeros((input_dim, 1))
        self.grad = np.zeros(self.weight.shape)
        self.lr = lr

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.input.shape != x.shape:
            raise ValueError(f"Expected shape {self.input.shape}, but input shape {x.shape}")
        self.input = x
        return np.dot(self.weight, self.input)

    def backward(self, dx: np.ndarray) -> np.ndarray:
        self.grad = np.dot(dx, self.input.reshape(1, -1))
        return np.dot(self.weight.T, dx)

    def update(self):
        self.weight -= self.grad * self.lr

    def __str__(self):
        return f"{self.get_layer_name()} : {self.input_dim} -> {self.output_dim}"


class ReLU(NetworkComponent):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def backward(self, dx: np.ndarray) -> np.ndarray:
        return np.where(dx > 0, 1, 0)


class Sigmoid(NetworkComponent):
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dx: np.ndarray) -> np.ndarray:
        return self.output * (1 - self.output)


class Softmax(NetworkComponent):
    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.exp(x) / np.sum(np.exp(x))
        return self.output

    def backward(self, dx: np.ndarray) -> np.ndarray:
        return self.output * (1 - self.output)
