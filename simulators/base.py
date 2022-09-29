import abc
import numpy as np
import torch

from utils.initial_design import sample_initial_inputs


class BaseSimulator(abc.ABC):
    """
    The base class for any simulator based on analytical expressions

    :param n_points: Number of data points in the discretized data set
    :param noise_sigma: The standard deviation of the noise
    :param n_inputs: number of inputs
    :param n_outputs: number of outputs
    """

    def __init__(self, n_points, noise_sigma, n_inputs, n_outputs):
        self.n_points = n_points
        self.noise_sigma = noise_sigma
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def query(self, x):
        """
        Query labels from the simulator
        :param x:
        :return: the output for x
        """
        y = self.mean(x) + self.stddev(x) * self.noise(x)
        if y.shape[1] == 1:
            y = y.reshape(-1)
        return y

    def sample_initial_data(self, n_samples, space_filling_design, seed=None):
        """
        Sample initial data
        :param n_samples: number of samples (data points) to sample
        :param space_filling_design: which space filling design to use
        :param seed:
        :return: the data set: x, y
        """
        #if seed is not None:
        #    np.random.seed(seed)

        # Sample training data
        x = np.array(sample_initial_inputs(n_samples, self.search_space(), method=space_filling_design, seed=seed))

        # Get labels from oracle
        y = self.query(x)

        x = torch.tensor(x, dtype=torch.float32) if not torch.is_tensor(x) else x
        if len(x.shape) == 1:
            x = x.view(-1, 1)
        y = torch.tensor(y, dtype=torch.float32) if not torch.is_tensor(y) else y
        if self.n_outputs == 1:
            y = y.squeeze(-1)

        return x, y

    @abc.abstractmethod
    def mean(self, x):
        """
        The mean of the simulator
        :param x:
        :return:
        """
        pass

    @abc.abstractmethod
    def stddev(self, x):
        """
        The standard deviation of the simulator
        :param x:
        :return:
        """
        pass

    @abc.abstractmethod
    def noise(self, x):
        """
        The noise of the simulator
        :param x:
        :return:
        """
        pass

    @abc.abstractmethod
    def search_space(self):
        """
        The search space of the simulator. Remember to always have a feature dimension
        :return:
        """
        pass
