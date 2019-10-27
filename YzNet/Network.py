import os
import re
from typing import List, Type, Dict, Union

import matplotlib.pyplot as plt

from YzNet.Components import *
from YzNet.LossFunctions import *


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

    def summary(self) -> None:
        print("=" * 10 + "Network Summary" + "=" * 10)
        for l in self.layers:
            print(l)

    def load_model(self, file_path: str = "./model.txt") -> None:
        file = open(file_path, mode="r", encoding="utf-8")
        str2class: Dict[str, Type[Union[Neuron, Sigmoid, ReLU, Softmax]]] = {
            Neuron.get_layer_name(): Neuron,
            Sigmoid.get_layer_name(): Sigmoid,
            ReLU.get_layer_name(): ReLU,
            Softmax.get_layer_name(): Softmax
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
