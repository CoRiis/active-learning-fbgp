import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import gpytorch
import torch

import pyro
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
import pyro.contrib.gp as gp

from sklearn.neighbors import KernelDensity

import models.exact_gp
from models.model import AbstractModel


class BaseFBGP(AbstractModel):
    """
    Base class for an Fully Bayesian GP (FBGP).
    Contains:
    - Prediction
    - Loss functions
    - Fitting procedure

    """

    def forward(self, x):
        """
        Function that feeds the data through the model.
        This function is dependent on the model.

        :param x: input data
        """
        raise NotImplementedError

    def predict(self, dataloader):
        """
        Function that predicts the label on x.

        :param x: input data
        """

        # Extract samples from posterior
        posterior_samples = self.mcmc.get_samples()

        if "covar_module.base_kernel.lengthscale_prior" in posterior_samples.keys():
            del posterior_samples["covar_module.base_kernel.lengthscale_prior"]
            del posterior_samples['likelihood.noise_prior']

        def get_mode_from_kde2d(data):
            kde = KernelDensity(kernel='gaussian', bandwidth=0.3)
            kde.fit(data)
            logprob = kde.score_samples(data)
            return data.iloc[np.argmax(logprob)].values.tolist()

        for l in range(posterior_samples['kernel.lengthscale'].shape[1]):
            key = 'lengthscale' + str(l)
            posterior_samples[key] = posterior_samples['kernel.lengthscale'][:, l]
        del posterior_samples['kernel.lengthscale']
        posterior_samples['noise0'] = posterior_samples['noise']
        del posterior_samples['noise']

        modes = get_mode_from_kde2d(pd.DataFrame(posterior_samples))

        # Make a ExactGPModel so we predict with that one
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        pred_model = models.exact_gp.ExactGPModel(self.train_x, self.train_y, self.gpytorch_likelihood,
                                                  mean_type='zero', covar_type='RBF_NoScale', ard=True)
        # NB: Make sure that gp.ExactGPModel has zero_mean and outputscale=1
        pred_model.covar_module.register_prior("lengthscale_prior", self.length_prior, "lengthscale")
        pred_model.likelihood.register_prior("noise_prior", self.gpytorch_likelihood.noise_prior, "noise")

        if self.args.predict_mcmc == "mode":
            pred_model.covar_module.lengthscale = torch.tensor(modes[:-1])
            pred_model.likelihood.noise = torch.tensor(modes[-1])
            self.pred_model = pred_model
            output = pred_model.predict(dataloader)
        elif self.args.predict_mcmc == 'posterior':
            raise NotImplementedError
        elif self.args.predict_mcmc == 'moments':
            pred_model.covar_module.lengthscale = torch.tensor(modes[:-1])
            pred_model.likelihood.noise = torch.tensor(modes[-1])
            self.pred_model = pred_model
            if self.batch_model is None:
                self.set_batch_model()

            # Now, we will do predictions with batch_model
            with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.cg_tolerance(0.01):
                batches = False if type(dataloader) == tuple else True
                if batches:
                    raise NotImplementedError
                else:
                    x, _ = dataloader
                    with torch.no_grad(), gpytorch.settings.fast_pred_var():
                        n_mcmc_samples = self.args.num_chains * self.args.num_samples
                        expanded_x = x.repeat(n_mcmc_samples, 1, 1)
                        predictions_f = self.batch_model(expanded_x)
                        predictions_y = self.batch_model.likelihood(predictions_f)

            output = {'predictions': predictions_y,
                      'mean': predictions_y.mean,
                      'stddev': predictions_y.stddev.detach(),
                      # 'stddev': predictions_f.stddev.detach()
                      }
        else:
            raise NotImplementedError

        return output

    def loss_func(self):
        """
        Function that return the loss function.
        For a GP w/ exact inference the loss function is given by the ExactMarginalLogLikelihood
        """
        return None

    def fit(self, train_data, args=None, debug=False, initialization=None):
        """
        Function that fits (train) the model on the data (x,y).

        :param train_data: tuple with (features / input data, label / output data)
        :param debug:
        :param initialization:
        :param args: arguments
        """

        pyro.clear_param_store()
        kernel = NUTS(self.gpr.model, target_accept_prob=0.8, use_multinomial_sampling=False)
        self.mcmc = MCMC(kernel, num_samples=args.num_samples, warmup_steps=args.warmup_steps,
                         num_chains=args.num_chains, disable_progbar=True, mp_context="spawn")
        self.mcmc.run()
        self.mcmc_samples = self.mcmc.get_samples()

        final_loss, losses = -1, np.linspace(-1, 1, 10)
        return final_loss, losses, None

    def set_batch_model(self):
        # First rename posterior_samples
        posterior_samples = self.mcmc.get_samples()
        posterior_samples["covar_module.lengthscale_prior"] = \
            posterior_samples['kernel.lengthscale'].view(-1, 1,self.train_x.shape[1])
        posterior_samples["likelihood.noise_prior"] = posterior_samples['noise'].view(-1, 1)

        # Turn model into batch model in order to draw mean functions
        likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
        batch_model = models.exact_gp.ExactGPModel(self.train_x, self.train_y, likelihood,
                                                  mean_type='zero', covar_type='RBF_NoScale', ard=True)

        # Add priors to turn model into batch model afterwards
        batch_model.covar_module.register_prior("lengthscale_prior", self.length_prior, "lengthscale")
        batch_model.likelihood.register_prior("noise_prior", self.noise_prior, "noise")
        # Manual thinning, only take every 10th sample
        thinning = 10
        total_samples = self.args.num_chains * self.args.num_samples
        thinning_index = np.linspace(0, total_samples - thinning, total_samples // thinning)
        posterior_samples['kernel.lengthscale'] = posterior_samples['kernel.lengthscale'][thinning_index]
        posterior_samples['noise'] = posterior_samples['noise'][thinning_index]
        posterior_samples['covar_module.lengthscale_prior'] = posterior_samples['covar_module.lengthscale_prior'][
                                                              thinning_index]
        posterior_samples['likelihood.noise_prior'] = posterior_samples['likelihood.noise_prior'][thinning_index]
        batch_model.pyro_load_from_samples(posterior_samples)

        batch_model.eval()
        self.batch_model = batch_model
        self.batch_mll = gpytorch.mlls.ExactMarginalLogLikelihood(batch_model.likelihood, batch_model)
