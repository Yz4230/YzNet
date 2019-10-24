import numpy as np
from Network import Neuron, MSE, ReLU, NeuralNetwork
import matplotlib.pyplot as plt

INPUT_DIM = 2
OUTPUT_DIM = 1
NUM_DATA = 4
LR = 0.001
A = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]])
B = np.array([[[0]], [[0]], [[0]], [[1]]])
print(f"{A.shape}, {B.shape}")

nn = NeuralNetwork()

layers = [Neuron(INPUT_DIM, 3, LR),
          ReLU(),
          Neuron(3, 3, LR),
          Neuron(3, OUTPUT_DIM, LR)]

for i in range(100):
    loss_sum = 0
    for a, b in zip(A, B):
        out = layers[0].forward(a)
        for l in layers[1:]:
            out = l.forward(out)
        loss = MSE.forward(out, b)
        loss_sum += loss
        grad = MSE.backward(out, b)
        grad = layers[-1].backward(grad)
        for l in reversed(layers[:-1]):
            grad = l.backward(grad)
        for l in layers:
            if isinstance(l, Neuron):
                l.update()
    loss_epoch.append(loss_sum / NUM_DATA)
    print(f"Loss = {loss_epoch[-1]}")
plt.plot(loss_epoch)
plt.show()
