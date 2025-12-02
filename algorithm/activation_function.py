import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @abstractmethod
    def forward(self, z: NDArray[float]) -> NDArray[float]:
        pass

class ReLU(ActivationFunction):

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        return np.maximum(0, z)

class Sigmoid(ActivationFunction):

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        return 1 / (1 + np.exp(-z))

class Tanh(ActivationFunction):

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        return np.tanh(z)

class Softmax(ActivationFunction):

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)