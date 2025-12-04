import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod


class ActivationFunction(ABC):

    """
    Abstract base class for activation functions.

    Methods
    -------
    forward(z: NDArray[float]) -> NDArray[float]
        Applies the activation function to the input.
    """

    @abstractmethod
    def forward(self, z: NDArray[float]) -> NDArray[float]:

        """
        Applies the activation function to the input data.

        Parameters
        ----------
        z : NDArray[float]
            An array with the pre-activation values (weighted sum + bias).

        Returns
        -------
        NDArray[float]
            An array with the activated values.
        """

        pass


class ReLU(ActivationFunction):

    """
    Rectified Linear Unit activation function.

    Formula: ``f(z)=max(0,z)``, with output range
    ``[0,+∞)``

    Methods
    -------
    forward(z: NDArray[float]) -> NDArray[float]
        Applies the activation function to the input.
    """

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        return np.maximum(0, z)


class Sigmoid(ActivationFunction):

    """
    Sigmoid activation function.

    Formula: ``f(z)=1/(1+e^(-z))``, with output
    range ``(0,1)``

    Methods
    -------
    forward(z: NDArray[float]) -> NDArray[float]
        Applies the activation function to the input.
    """

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        return 1 / (1 + np.exp(-z))


class Tanh(ActivationFunction):

    """
    Hyperbolic tangent activation function.

    Formula: ``f(z)=tanh(z)``, with output
    range ``(-1,1)``

    Methods
    -------
    forward(z: NDArray[float]) -> NDArray[float]
        Applies the activation function to the input.
    """

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        return np.tanh(z)


class Softmax(ActivationFunction):

    """
    Softmax activation function.

    Formula: ``f(z_i)=e^(z_i)/Σe^(z_j)``, with
    output range ``(0,1)``, summing to 1.

    Methods
    -------
    forward(z: NDArray[float]) -> NDArray[float]
        Applies the activation function to the input.

    Notes
    -----
    Used for the output layer of neural networks.
    """

    def forward(self, z: NDArray[float]) -> NDArray[float]:
        exp_z: NDArray[float] = np.exp(z - np.max(z, axis=-1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
