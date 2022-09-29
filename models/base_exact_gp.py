# Base function for Gaussian Processes w/ exact inference
import gpytorch
import torch
import abc
import numpy as np

from utils.optimizer import get_optimizer
from models.model import AbstractModel


class BaseExactGPModel(AbstractModel, gpytorch.models.ExactGP):
    """
    Base class for an ExactGP.
    Contains:
    - Prediction
    - Loss functions
    - Fitting procedure

    """

    @abc.abstractmethod
    def forward(self, x):
        raise NotImplementedError

    def predict(self, dataloader):
        with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.cg_tolerance(0.01):
            self.eval()
            batches = False if type(dataloader) == tuple else True
            if batches:
                raise NotImplementedError
            else:
                x, _ = dataloader
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    predictions_f = self(x)
                    predictions_y = self.likelihood(predictions_f)

        if type(self.covar_module) == gpytorch.kernels.periodic_kernel.PeriodicKernel:
            stddev_f = torch.sqrt(predictions_f.covariance_matrix.detach().diag())
            stddev_y = torch.sqrt(predictions_y.covariance_matrix.detach().diag())
        else:
            stddev_f = predictions_f.stddev.detach()
            stddev_y = predictions_y.stddev.detach()

        output = {'predictions': predictions_y,
                  'mean': predictions_y.mean,
                  'stddev': stddev_y,
                  'stddev_f': stddev_f
                  }
        return output

    def loss_func(self):
        return gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

    def fit(self, train_data, args=None, debug=False, initialization=None, test_data=None):
        x, y = train_data

        # Settings
        n_runs = args.n_runs
        training_iter = args.n_epochs
        opt, scheduler = get_optimizer(args, self, num_data=y.size(0))
        optimizer = opt[0]
        ngd_optimizer = opt[1]

        # Fit the model
        min_loss = 10e10
        mll = self.loss_func()
        self.train()
        for run in range(n_runs):
            tmp_losses = []
            tmp_lr = []
            outputscales = torch.empty([training_iter, args.outputs])
            lengthscales = torch.empty([training_iter, args.outputs, x.shape[1]])
            means = torch.empty([training_iter, args.outputs])
            noises = torch.empty([training_iter, args.outputs])
            noises2 = torch.empty([training_iter])

            for i in range(training_iter):
                # Zero gradients from previous iteration
                if ngd_optimizer is not None:
                    ngd_optimizer.zero_grad()
                optimizer.zero_grad()
                with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.cg_tolerance(0.01): #, gpytorch.settings.cholesky_jitter(1e-3):
                    output = self(x)
                    loss = -mll(output, y)
                    loss.backward()
                if ngd_optimizer is not None:
                    ngd_optimizer.step()
                optimizer.step()

                # Learning rate scheduler
                scheduler.step(loss)
                tmp_losses.append(loss.item())

                if debug:
                    for param_group in optimizer.param_groups:
                        tmp_lr.append(param_group['lr'])

                if i > 15:
                    # Stop, if the loss doesn't change within 15 iterations (precision until 4 decimal)
                    tmp_losses_np_round = np.round(np.array(tmp_losses[-15:]), 4)
                    if np.all(tmp_losses_np_round == tmp_losses_np_round[-1]):
                        break

                # Save the best model
                if tmp_losses[-1] < min_loss:
                    min_loss = tmp_losses[-1]
                    losses = list(tmp_losses)

        outs = {'outputscales': outputscales,
                'lengthscales': lengthscales,
                'means': means,
                'noises': noises,
                'noises2': noises2}

        return losses[-1], losses, outs

    def mean(self, mean_type: str):
        """
        Species the mean module of the GP
        :param mean_type [zero, constant, linear]
        :return: gpytorch.means
        """

        if mean_type == 'zero':
            mean_module = gpytorch.means.ZeroMean()
        elif mean_type == 'constant':
            mean_module = gpytorch.means.ConstantMean()
        elif mean_type == 'linear':
            mean_module = gpytorch.means.LinearMean(self.train_inputs[0].shape[1])
        else:
            raise NotImplementedError(f"The {mean_type} is not implemented.")

        return mean_module

    def covar(self, covar_type: str, ard: bool, prior: bool):
        """
        Species the covariance module (kernel) of the GP
        :param covar_type:
        :param ard: Automatic Relevance Determination
        :return: gpytorch.kernels
        """

        ard_num_dims = self.train_inputs[0].shape[1] if ard else None

        if covar_type == 'RBF':
            if prior == "lalprior":
                # Lalchand2019
                lengthscale_prior = gpytorch.priors.LogNormalPrior(0, torch.sqrt(torch.tensor([3])))
                outputscale_prior = gpytorch.priors.LogNormalPrior(0, torch.sqrt(torch.tensor([3])))
                covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=ard_num_dims,
                        lengthscale_prior=lengthscale_prior
                    ),
                    outputscale_prior=outputscale_prior
                )
            elif prior == "boprior":
                # BoTorch and GPyTorch
                # https://botorch.org/api/_modules/botorch/models/gp_regression.html#SingleTaskGP
                # https://docs.gpytorch.ai/en/stable/examples/00_Basic_Usage/Hyperparameters.html#Priors
                lengthscale_prior = gpytorch.priors.GammaPrior(3.0, 6.0)
                outputscale_prior = gpytorch.priors.GammaPrior(2.0, 0.15)
                covar_module = gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(
                        ard_num_dims=ard_num_dims,
                        lengthscale_prior=lengthscale_prior
                    ),
                    outputscale_prior=outputscale_prior
                )
            else:
                covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims))
        elif covar_type == 'RBF_NoScale':
            covar_module = gpytorch.kernels.RBFKernel(ard_num_dims=ard_num_dims)
        else:
            raise NotImplementedError(f"The '{covar_type}' is not implemented.")

        return covar_module
