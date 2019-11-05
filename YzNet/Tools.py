import numpy as np


def onehot(x: np.ndarray) -> np.ndarray:
    n_classes = len(np.unique(x))
    ret: np.ndarray = np.eye(n_classes)[x.flatten()]
    return ret.reshape([x.shape[0], x.shape[1] * n_classes, 1]).astype(np.int32)
