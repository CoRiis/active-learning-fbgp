import abc
import torch


class AbstractModel(abc.ABC):
    """
    Abstract class for a model

    """
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Function that feeds the data through the model.
        This function is dependent on the model.

        :param x: input data
        :return y: output of model
        """
        raise NotImplementedError

    @abc.abstractmethod
    def predict(self, dataloader):
        """
        Function that predicts the label on x.
        This function should be able to handle a dataloader that is either a tuple (x,y) or a pytorch dataloader

        :param dataloader: either a tuple with (x,y) or a pytorch dataloader
        :return output: a dictionary w/
            {
             predictions: predictions,
             mean: mean of the predictions,
             stddev: std. deviation of the predictions,
             stddev_f: std. deviation of the mean
             }
        """
        raise NotImplementedError

    @abc.abstractmethod
    def loss_func(self):
        """
        Function that return the loss function.

        For example, for a GP w/ exact inference the loss function is given by the ExactMarginalLogLikelihood.
        Thus, this function should return gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        :return loss function
        """
        raise NotImplementedError

    @abc.abstractmethod
    def fit(self, train_data, args=None, debug=False, initialization=None, test_data=None):
        """
        Function that fits (train) the model on the data (x,y).

        :param train_data: tuple with (features / input data, label / output data)
        :param args: arguments
        :param debug: bool
        :param initialization: integer for initialization
        :param test_data: tuple with test data (x_test, y_test)
        :return three outputs: the last loss, the list with losses, and a dictionary with misc. variables.
        """
        raise NotImplementedError
