from abc import abstractmethod, ABCMeta
from typing import List, Type, Dict, Union
import re
import matplotlib.pyplot as plt
import numpy as np
import os


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


class Neuron(NetworkComponent):
    def __init__(self, input_dim: int, output_dim: int, lr: float = 0.001):
        self.input_dim = input_dim
        self.output_dim = output_dim
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

    def __str__(self):
        return f"{self.get_layer_name()} : input({self.input_dim}) -> output({self.output_dim})"


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
    layers: List[NetworkComponent]
    history: List[np.float64]

    def __init__(self):
        self.layers = []
        self.history = []

    def add_layer(self, layer: NetworkComponent) -> None:
        self.layers.append(layer)

    def learn(self, x: np.ndarray, y: np.ndarray, epoch: int) -> list:
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        feed_forward = self.layers[0].forward(x)
        for l in self.layers[1:]:
            feed_forward = l.forward(feed_forward)
        return feed_forward

    def plot_loss(self) -> None:
        plt.plot(self.history)
        plt.show()

    def load_model(self, file_path: str = "./model.txt") -> None:
        file = open(file_path, mode="r", encoding="utf-8")
        str2class: Dict[str, Type[Union[Neuron, Sigmoid, ReLU]]] = {
            Neuron.get_layer_name(): Neuron,
            Sigmoid.get_layer_name(): Sigmoid,
            ReLU.get_layer_name(): ReLU
        }
        pattern_option = re.compile(r"([^ ,\n]+)")
        for l in file:
            re_res: List[str] = pattern_option.findall(l)
            layer_class = str2class.get(re_res[1])
            if layer_class is not None:
                if layer_class == Neuron:
                    self.layers.append(layer_class(int(re_res[2]), int(re_res[3]), float(re_res[4])))
                else:
                    self.layers.append(layer_class())
        file.close()

    def load_weight(self, file_dir: str = "./weight") -> None:
        if file_dir[-1] != "/":
            file_dir += "/"
        for n_layer, l in enumerate(self.layers):
            if isinstance(l, Neuron):
                file_path = f"{file_dir}{n_layer}_{l.get_layer_name()}.npy"
                l.weight = np.load(file_path)

    def save_model(self, file_path: str = "./model.txt") -> None:
        file = open(file_path, mode="w", encoding="utf-8")
        layer_detail: List[str] = []
        for n_layer, l in enumerate(self.layers):
            detail = f"{n_layer}, {l.get_layer_name()}, "
            if isinstance(l, Neuron):
                detail += f"{l.input_dim}, {l.output_dim}, {l.lr}, "
            layer_detail.append(detail)
        file.writelines("\n".join(layer_detail))
        file.close()

    def save_weight(self, file_dir: str = "./weight") -> None:
        if not os.path.exists(file_dir):
            os.mkdir(file_dir)
        if file_dir[-1] != "/":
            file_dir += "/"
        for n_layer, l in enumerate(self.layers):
            if isinstance(l, Neuron):
                file_path = f"{file_dir}{n_layer}_{l.get_layer_name()}.npy"
                np.save(file_path, l.weight)
