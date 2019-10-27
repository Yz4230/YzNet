import numpy as np


def onehot(x: np.ndarray) -> np.ndarray:
    n_classes = np.max(x) - np.min(x) + 1
    ret = np.zeros((*x.shape[:-2], n_classes, 1))
    ret: np.ndarray = ret.reshape(ret.shape[:-1])
    for xi, yi in zip(x.reshape(x.shape[:-1]), ret):
        yi[xi] = 1
    a = np.array(list(range(10)))
    return ret.reshape(*ret.shape, 1).astype(np.int32)
