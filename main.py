from YzNet.Components import *
from YzNet.Network import NeuralNetwork
from YzNet.Tools import onehot

INPUT_DIM = 2
OUTPUT_DIM = 1
NUM_DATA = 4
LR = 0.001
A = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]])
B = np.array([[[0]], [[0]], [[0]], [[1]]])
print(f"{A.shape}, {B.shape}")
print(onehot(B))
