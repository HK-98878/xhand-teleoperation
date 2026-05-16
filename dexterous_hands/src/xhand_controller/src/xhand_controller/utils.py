from typing import Sequence, Union

import numpy as np


def sequence_smoother(origin_sequence: np.ndarray, T: float, alpha: float = 0.1, beta: float = 0.01) -> np.ndarray:
    """
    Smooths a given sequence using MSE algorithm.

    Parameters:
    origin_sequence (np.ndarray): The original sequence to be smoothed.
    T (float): The time constant for the smoothing process.
    alpha (float): The smoothing parameter for the first-order difference.
    beta (float): The smoothing parameter for the second-order difference.
    Returns:
    np.ndarray: The smoothed sequence.
    """
    origin_sequence = origin_sequence[::-1]
    length = len(origin_sequence)
    A = np.eye(length)
    B = np.eye(length)
    for i in range(length):
        for j in range(length):
            if i == j - 1:
                A[i, j] = -1
                B[i, j] = -2
            if i == j - 2:
                B[i, j] = 1
    A[-1] = 0
    B[-2:] = 0
    Q = np.eye(length) + alpha / T * np.matmul(A.transpose(), A) + beta / (T**2) * np.matmul(B.transpose(), B)
    Q_pinv = np.linalg.pinv(Q)
    x = np.matmul(Q_pinv, origin_sequence)
    return x[::-1]


def smoother(
    origin_points: Union[Sequence, np.ndarray],
    T: float,
    alpha: float = 0.1,
    beta: float = 0.01,
) -> float:
    """
    Smooths a sequence of points using the specified parameters.

    Args:
        origin_points (list or array-like): The original sequence of points to be smoothed.
        T (float): The time parameter for the smoothing algorithm.
        alpha (float): The alpha parameter for the smoothing algorithm.
        beta (float): The beta parameter for the smoothing algorithm.

    Returns:
        float: The last value of the smoothed sequence.
    """
    if not isinstance(origin_points, np.ndarray):
        origin_points = np.ndarray(origin_points)
        return sequence_smoother(origin_sequence=origin_points, alpha=alpha, beta=beta, T=T)[-1].item()
    else:
        return sequence_smoother(origin_sequence=origin_points, alpha=alpha, beta=beta, T=T)[-1]


class Smoother:
    """
    A class used to smooth data using a specified time constant, alpha, and beta parameters.

    Attributes:
        alpha (float): The pos parameter for the controller.
        beta (float): The velocity parameter for the controller.
        history_len (int): The length of the history to maintain.
        _history (np.ndarray or None): The history of the data.

    Methods:
        __init__(T: float, alpha: float = 0.1, beta: float = 0.01, history_len: int = 5):
            Initializes the Smoother with the given parameters.
        forward(data):
            Smooths the input data using the specified parameters.
        __call__(data):
            Alias for the forward method.
    """

    def __init__(self, T: float, alpha: float = 0.1, beta: float = 0.01, history_len: int = 5):
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self._history = None
        self.history_len = history_len
        self._log_data = {}

    def forward(self, data):
        if np.isnan(data).any():
            return data
        if not isinstance(data, np.ndarray):
            data = np.array(data).reshape(-1)
        self.log(data=data)

        if self._history is None:
            self._history = data[None, ...]
            self._history = np.vstack([self._history] * self.history_len)
        else:
            self._history = np.vstack([self._history, data[None, ...]])
        if len(self._history) > self.history_len:
            self._history = self._history[-self.history_len :, ...]
            data = sequence_smoother(self._history, T=self.T, alpha=self.alpha, beta=self.beta)[-1].reshape(*data.shape)
        self.log(filtered_data=data)

        return data

    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self._log_data:
                self._log_data[key] = []
            self._log_data[key].append(value)

    __call__ = forward
