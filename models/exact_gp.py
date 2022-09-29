# Gaussian Processes w/ exact inference
import gpytorch
from models.base_exact_gp import BaseExactGPModel


class ExactGPModel(BaseExactGPModel):
    """
    A single task (output) GP model w/ exact inference.

    """
    def __init__(self, train_x, train_y, likelihood, mean_type='zero',
                 covar_type='RBF', ard=True, prior=False):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = self.mean(mean_type)
        self.covar_module = self.covar(covar_type=covar_type, ard=ard, prior=prior)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
