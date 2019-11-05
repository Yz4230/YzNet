from YzNet.Components import *
from YzNet.Network import NeuralNetwork
from YzNet.Tools import onehot

INPUT_DIM = 4
OUTPUT_DIM = 2
NUM_DATA = 4
LR = 0.001
A = np.array([[[0], [0]], [[1], [0]], [[0], [1]], [[1], [1]]])
B = np.array([[[0]], [[0]], [[0]], [[1]]])
A = onehot(A)
B = onehot(B)
nn = NeuralNetwork()

nn.add_layer(Neuron(INPUT_DIM, 4))
nn.add_layer(Neuron(4, 4))
nn.add_layer(Neuron(4, OUTPUT_DIM))
nn.add_layer(Softmax())

nn.learn(A, B, 5000)
nn.plot_loss()
nn.save_model()
nn.save_weight()
print(*[nn.predict(a) for a in A], sep="\n")
