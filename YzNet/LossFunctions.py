from abc import abstractmethod, ABCMeta

import numpy as np


class ErrorFunction(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def forward(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        pass

    @classmethod
    @abstractmethod
    def backward(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        pass


class MSE(ErrorFunction):
    @classmethod
    def forward(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        return np.mean(np.square(y.flatten() - x.flatten())) / 2

    @classmethod
    def backward(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.shape != y.shape:
            raise ValueError(f"Invalid shape, {x.shape} != {y.shape}")
        return (x.reshape(-1, 1) - y.reshape(-1, 1)) / x.size


class CrossEntropy(ErrorFunction):
    @classmethod
    def forward(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        return -np.sum(y * np.log(x))

    @classmethod
    def backward(cls, x: np.ndarray, y: np.ndarray) -> np.float64:
        return -y / x
