# Utility functions for Gaussian Processes (GP)
import torch
import gpytorch
import numpy as np
import pyro
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood

import utils.metrics as metrics
from models.pyro_gp import FBGP


def get_likelihood(args):
    """
    Returns the likelihood

    :param args: arguments
    """

    likelihood = GaussianLikelihood()
    likelihood.noise_constraint = gpytorch.constraints.Positive()
    set_prior_on_likelihood(args, likelihood)

    return likelihood


def set_prior_on_likelihood(args, likelihood):
    """
    Set a prior on the likelihood
    :param args: arguments
    :param likelihood: gpytorch likelihood
    :return: add a prior on the likelihood
    """

    # Create a GaussianLikelihood with a normal prior for the noise
    if args.covar_prior == "lalprior":
        noise_prior = gpytorch.priors.LogNormalPrior(loc=0, scale=torch.sqrt(torch.tensor([3])))
    elif args.covar_prior == "boprior":
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05)
    else:
        return None

    likelihood.noise_prior = noise_prior
    likelihood.noise = 0.1

    return None


def get_model(args, data, likelihood):
    """
    Returns the model. All models will a specific likelihood defined inside the model class.

    :param args: arguments
    :param train_x: torch tensor with training data x (input)
    :param train_y: torch tensor with training data y (output)
    :param kernel: gpytorch kernel, e.g gpytorch.kernels.RBFKernel()
    :param likelihood: gpytorch likelihood, e.g. gpytorch.likelihoods.GaussianLikelihood()
    :param iteration: used to set priors TODO: Not tested
    :param opt_hypers: dict of parameters TODO: Not tested
    :param length_prior: prior to use in the PyroGP FBGP
    :param noise_prior: prior to use in  the PyroGP FBGP
    """

    train_x = data.train_trans.x
    train_y = data.train_trans.y

    """
    Fully Bayesian Gaussian Process
    Because of the connection between pyro and GPyTorch, the lengthscale is defined in FBGP() directly.
    """
    pyro_kernel = pyro.contrib.gp.kernels.RBF(input_dim=train_x.shape[1])
    noise_prior = gpytorch.priors.LogNormalPrior(loc=0, scale=1.73)

    # NB: Prior must be the same as likelihood for exact_prior (pyro only defined on positive values)
    model = FBGP(args, train_x, train_y, pyro_kernel, length_prior=None, noise_prior=noise_prior)
    model.gpytorch_likelihood = likelihood

    return model


def get_loss(args, model, num_data):
    """
    Returns the loss

    :param args: arguments
    :param model: model
    :param num_data: number of data points in the training set
    :returns: loss function used to train model
    """
    # Define loss ("Loss" for GPs - the marginal log likelihood)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.pred_model.likelihood, model.pred_model)
    return mll


def compute_loss(args, dataloader, predictions, lst_metrics=[], mll=None):
    """
    Validation of predictions: compares the labels in the test loader with the predictions.

    :param args: arguments
    :param dataloader: a data loader suited for either DeepGP (batches) or other GPs (non batches)
    :param predictions: predictions
    :param lst_metrics: list of metrics to calculate
    :param mll: marginal log likelihood (must be given together with mll in lst_metrics in order to calculate mll
    :returns: dictionary with losses
    """

    # Check if dataloader is a PyTorch dataloader or a tuple
    # If it is a PyTorch dataloader, it uses batches
    batches = False if type(dataloader) == tuple else True

    losses = {'nmll': -1}
    if batches:
        test_y = torch.tensor([])
        nmll_losses_valid_batch = []
        for idx, batch in enumerate(dataloader):
            _, y_batch = batch
            if 'mll' in lst_metrics and mll is not None and args.model_type not in ['ridge_reg', 'xgboost']:
                nmll_losses_valid_batch.append(torch.mean(-mll(predictions[idx], y_batch)).item())
        if 'mll' in lst_metrics and mll is not None and args.model_type not in ['ridge_reg', 'xgboost']:
            losses['nmll'] = np.mean(nmll_losses_valid_batch).item()
    else:
        test_x, test_y = dataloader
        if 'mll' in lst_metrics and mll is not None and args.model_type not in ['ridge_reg', 'xgboost']:
            losses['nmll'] = torch.mean(-mll(predictions, test_y)).item()

    if 'rmse' in lst_metrics:
        losses['rmse'] = metrics.rmse(preds=predictions.view(test_y.shape), targets=test_y)
    if 'mae' in lst_metrics:
        losses['mae'] = metrics.mae(preds=predictions.view(test_y.shape), targets=test_y)
    if 'rae' in lst_metrics:
        losses['rae'] = metrics.rea(preds=predictions.view(test_y.shape), targets=test_y)
    if 'rrse' in lst_metrics:
        losses['rrse'] = metrics.rrse(preds=predictions.view(test_y.shape), targets=test_y)

    return losses
