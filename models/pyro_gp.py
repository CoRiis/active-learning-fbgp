import pyro
from models.base_pyro_gp import BaseFBGP
import torch
import gpytorch


class FBGP(BaseFBGP):
    def __init__(self, args, train_x, train_y, kernel, length_prior, noise_prior):
        super().__init__()

        self.args = args
        self.batch_model = None
        self.mcmc_samples = None
        self.gpytorch_kernel = None
        self.gpytorch_likelihood = None

        # MCMC wants to pickle model, but cannot pickle class object. Thus, we must use this hacky method
        # to define the model as an object within the model
        self.gpr = pyro.contrib.gp.models.GPRegression(train_x, train_y, kernel)

        # Add training data to model
        self.train_x = train_x
        self.train_y = train_y

        # Place priors on GP covariance function parameters.
        self.length_prior = gpytorch.priors.LogNormalPrior(loc=0., scale=1.73)
        self.noise_prior = noise_prior
        self.gpr.kernel.lengthscale = pyro.nn.PyroSample(
            pyro.distributions.LogNormal(torch.tensor([0.]).repeat(train_x.shape[1]),
                                         torch.tensor([1.73]).repeat(train_x.shape[1])).to_event())
        self.gpr.noise = pyro.nn.PyroSample(noise_prior)
