# Base for Bayesian Acquisition Function (baf)
import abc
import torch
from scipy import integrate

from acquisition_functions.base_af import BaseAcquisitionFunction

SQRT_2PI = 2.5066282
TWO_PI_EXP = 17.0794684453
HALF_LOG_2PI = 0.9189385


class BaseBayesianAcquisitionFunction(BaseAcquisitionFunction):
    def __init__(self, model, candidate_points):
        super().__init__(model, candidate_points)

        if self.model.batch_model is None:
            model.set_batch_model()

        # Expand input space to predict with batch model
        cp = self.candidate_points
        cp = cp.view(-1, 1) if len(cp.shape) == 1 else cp
        thinning=10
        self.expanded_cp = cp.unsqueeze(0).repeat((self.model.args.num_samples * self.model.args.num_chains) // thinning, 1, 1)

        # Predict with batch model
        self.model.batch_model.eval()
        self.predictions = self.model.batch_model(self.expanded_cp)

    @abc.abstractmethod
    def evaluate(self, x):
        pass


class BALD(BaseBayesianAcquisitionFunction):
    def __init__(self, model, candidate_points):
        super().__init__(model, candidate_points)

    def evaluate(self, x=None):
        if x is None:
            expanded_test_x = self.expanded_cp
        else:
            x = x.view(-1, 1) if len(x.shape) == 1 else x
            expanded_test_x = x.unsqueeze(0).repeat(self.model.args.num_samples * self.model.args.num_chains, 1, 1)

        output = self.predictions
        mean_stddev_all = torch.mean(output.stddev.detach(), dim=0)
        entropy_expected = torch.log(mean_stddev_all)
        expected_entropy = torch.mean(torch.log(output.stddev.detach()), dim=0)
        bald = entropy_expected - expected_entropy
        return bald


class BQBC(BaseBayesianAcquisitionFunction):
    """
    Bayesian Query-by-Committee (B-QBC)
    """
    def __init__(self, model, candidate_points):
        super().__init__(model, candidate_points)

    def evaluate(self, x):
        output = self.predictions
        variance_of_means = torch.pow(torch.std(output.mean.detach(), dim=0), 2)
        return variance_of_means


class QBMGP(BaseBayesianAcquisitionFunction):
    """
    Query by Mixture of Gaussian Processes (QB-MGP)
    """
    def __init__(self, model, candidate_points):
        super().__init__(model, candidate_points)

    def evaluate(self, x):
        output = self.predictions
        mean_variance = torch.mean(torch.pow(output.stddev.detach(), 2), dim=0)
        variance_of_means = torch.pow(torch.std(output.mean.detach(), dim=0), 2)
        return mean_variance + variance_of_means


class BALM(BaseBayesianAcquisitionFunction):
    """
    Bayesian Active Learning MacKay (B-ALM).
    The Bayesian component comes from averaging over the models instead of using the mode.
    """
    def __init__(self, model, candidate_points):
        super().__init__(model, candidate_points)

    def evaluate(self, x):
        output = self.predictions
        mean_variance = torch.mean(torch.pow(output.stddev.detach(), 2), dim=0)
        return mean_variance
