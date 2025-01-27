import time

import gpytorch
from gpytorch.means import ConstantMean
from gpytorch.models import ExactGP
from gpytorch.distributions import MultivariateNormal
from botorch.models.gpytorch import GPyTorchModel
from botorch.posteriors.gpytorch import GPyTorchPosterior
from src.kernels import RBFKernelForConvexityConstraints
from src.samplers import TruncatedMVN
from src.utils import _match_batch_dims, _match_dtype
import torch

from typing import Any, Union, List
from torch import Tensor
from gpytorch import settings
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.utils.broadcasting import _mul_broadcast_shape
import numpy as np


class ConstrainedGPSEModel(ExactGP, GPyTorchModel):
    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self,
                 train_x: Tensor,
                 train_y: Tensor,
                 constrained_dims: List[int],
                 unconstrained_dims: List[int],
                 lengthscale_constraint=None,
                 lengthscale_hyperprior=None,
                 noise_constraint=None,
                 outputscale_constraint=None,
                 outputscale_hyperprior=None,
                 prev_trunc_samples=None,
                 prior_mean=0,):

        # check dimensions
        if train_y is not None:
            train_y = train_y.squeeze(-1)

        # constrain homoskedastic noise
        if noise_constraint is None:
            noise_constraint = gpytorch.constraints.GreaterThan(1e-6)

        # init likelihood
        likelihood = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=noise_constraint)

        # initialize model
        super(ConstrainedGPSEModel, self).__init__(train_x, train_y, likelihood)

        # track constrained dimensions
        self.constr_dims = constrained_dims
        self.unconstr_dims = unconstrained_dims

        # split kernel into spatio kernel and constrained spatio kernel K = K_c * K_s
        self.constrained_kernel = RBFKernelForConvexityConstraints(
            constrained_dims=self.constr_dims,
            ard_num_dims=len(self.constr_dims),
            active_dims=tuple(self.constr_dims),
            lengthscale_prior=lengthscale_hyperprior,
            lengthscale_constraint=lengthscale_constraint)

        self.unconstrained_kernel = gpytorch.kernels.RBFKernel(
            ard_num_dims=len(self.unconstr_dims),
            active_dims=tuple(self.unconstr_dims),
            lengthscale_prior=lengthscale_hyperprior,
            lengthscale_constraint=lengthscale_constraint)

        self.spatio_kernel = gpytorch.kernels.ScaleKernel(
            self.constrained_kernel,
            outpurtscale_prior=outputscale_hyperprior,
            outputscale_constraint=outputscale_constraint)

        # Initialize lengthscale and outputscale to mean of priors.
        if not self.unconstr_dims:
            if lengthscale_hyperprior is not None:
                self.spatio_kernel.base_kernel.lengthscale = lengthscale_hyperprior.mean
            if outputscale_hyperprior is not None:
                self.spatio_kernel.outputscale = outputscale_hyperprior.mean

            self.covar_module = self.spatio_kernel
        else:
            if lengthscale_hyperprior is not None:
                self.spatio_kernel.base_kernel.lengthscale = lengthscale_hyperprior.mean
                self.unconstrained_kernel.lengthscale = lengthscale_hyperprior.mean
            if outputscale_hyperprior is not None:
                self.spatio_kernel.outputscale = outputscale_hyperprior.mean
            self.covar_module = self.spatio_kernel * self.unconstrained_kernel

        # Initialize mean
        self.mean_module = gpytorch.means.ConstantMean()
        if prior_mean != 0:
            self.mean_module.initialize(constant=prior_mean)
        self.mean_module.constant.requires_grad = False

        # factors, that do not need to be recomputed (AGRELL2019)
        self.C = None
        self.U = None
        self.v1 = None
        self.A1 = None
        self.B1 = None
        self.L1 = None
        self.L = None
        self.covar_train_train_noisy = None
        self.covar_train_ddxv = None
        self.covar_ddxv_ddxv = None
        self.centered_mean = None
        self.centered_mean_nobatch = None
        self.proposal_set = None
        self.VOPs = False

        self.location_noise = 1e-6  # numerical stability -> lowering the probability of hard constraints
        self.bounds = [-np.inf, 0]  # numpy since sampling algorithm needs numpy bounds
        self.prev_trunc_samples = prev_trunc_samples
        self.prev_post_samples = None
        self.posterior_distribution = None

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def posterior(self, X: Tensor, observation_noise: Union[bool, Tensor] = False, **kwargs: Any
                  ) -> GPyTorchPosterior:

        if 'evaluate_constrained' in kwargs:
            if kwargs['evaluate_constrained'] is True:
                # get obseration points
                self.VOPs = kwargs['virtual_oberservation_points']
                if not isinstance(self.VOPs, Tensor):
                    raise RuntimeError("Virtual observations must be a tensor!")

                n_samples = kwargs['nr_samples']

                # check if batches are used
                batch_mode = len(X.shape) == 3

                # get joint distribution of training and testing
                full_mean, full_covar = self.get_joint_distribution(X)

                # assign mean and covariances
                train_mean = full_mean[..., :self.num_train]
                test_mean = full_mean[..., self.num_train:]

                covar_train_test = full_covar[..., :self.num_train, self.num_train:].evaluate()
                covar_test_test = full_covar[..., self.num_train:, self.num_train:].evaluate()

                # calculate training covar (can reuse)
                measurement_noise = self.likelihood.noise_covar.noise
                if not isinstance(self.covar_train_train_noisy, Tensor):
                    covar_train_train = full_covar[..., :self.num_train, :self.num_train].evaluate()
                    covar_XX_noisy = torch.add(covar_train_train, torch.eye(self.num_train) * measurement_noise)
                    self.covar_train_train_noisy = covar_XX_noisy[0, ...] if batch_mode else covar_XX_noisy

                # match dimensions
                X, training_inputs, training_targets = _match_batch_dims(X, self.train_inputs[0], self.train_targets)

                self.centered_mean = training_targets - train_mean
                self.centered_mean_nobatch = self.centered_mean[0, ...] if batch_mode else self.centered_mean

                # calculate and construct the derivative covar matrices
                covar_train_ddxv, covar_ddxv_ddxv, covar_test_ddxv = self.get_derivative_covars(X, training_inputs)

                # calculate factors (Agrell 2019)
                L, v1, A1, B1, L1 = self.get_factors()
                X, L, v1, A1, B1, L1 = _match_batch_dims(X, L, v1, A1, B1, L1)  # match batch dimensions
                L_transpose = L.transpose(dim0=-2, dim1=-1)

                v2 = torch.triangular_solve(covar_train_test, L, upper=False).solution
                A2 = torch.triangular_solve(v2, L_transpose, upper=True).solution.transpose(dim0=-2, dim1=-1)

                B2 = covar_test_test - v2.transpose(dim0=-2, dim1=-1) @ v2
                B3 = covar_test_ddxv - v2.transpose(dim0=-2, dim1=-1) @ v1
                L1_transpose = L1.transpose(dim0=-2, dim1=-1)  # for numerical stability

                v3 = torch.triangular_solve(B3.transpose(dim0=-2, dim1=-1), L1, upper=False).solution

                A = torch.triangular_solve(v3, L1_transpose, upper=True).solution.transpose(dim0=-2, dim1=-1)
                B = A2 - A @ A1

                # mu_s and and cov_s only needed for sampling - samples can be used for the whole batch since
                # observations and training do not change
                if batch_mode:
                    mu_s = A1[0, ...] @ self.centered_mean_nobatch
                    cov_s = B1[0, ...]
                    self.centered_mean = self.centered_mean.unsqueeze(-1)
                    test_mean = test_mean.unsqueeze(-1)
                else:
                    mu_s = A1 @ self.centered_mean
                    cov_s = B1

                sigma = B2 - v3.transpose(dim0=-2, dim1=-1) @ v3
                Q = psd_safe_cholesky(sigma, max_tries=6)  # for numerical stability

                # reuse samples!
                if self.C is None:
                    C = self.sample_from_truncated_norm(mu_s, cov_s, n_samples=n_samples,
                                                        sampling_method='minimax_tilting')
                    self.C = C
                    self.prev_trunc_samples = C[:, -100:]
                else:
                    C = self.C

                # reshape to batches if needed
                if batch_mode:
                    _, C = _match_batch_dims(X, C)  # reuse samples for whole batch

                # draw samples (dim(x_test) x n_samples) from multivariate standard distribution
                m = torch.distributions.MultivariateNormal(torch.zeros(Q.shape[-1]), torch.eye(Q.shape[-1]))
                # torch.manual_seed(356997) # seed showed to be not beneficial as it was influencing the result
                U = m.rsample(sample_shape=torch.Size([C.shape[-1]])).T

                # calculate mean
                prior_test_mean = (test_mean + B @ self.centered_mean)
                if batch_mode:
                    prior_test_mean = torch.tile(prior_test_mean, (1, 1, C.shape[-1]))
                else:
                    prior_test_mean = torch.tile(prior_test_mean.unsqueeze(-1), (1, C.shape[-1]))

                # convert samples to type of the input
                C, U = _match_dtype(X, C, U)

                # approximate posterior f*| X, Y, C
                fs_sim = prior_test_mean + A @ C + Q @ U

                post_mean = fs_sim.mean(axis=-1)
                post_cov = fs_sim.var(axis=-1)

                # store random samples from the posterior
                if not batch_mode:
                    self.prev_post_samples = fs_sim

                # convert to diagonal matrix (b x len(test) x len(test))
                diagmat_post_cov = torch.diag_embed(post_cov, offset=0, dim1=-2, dim2=-1)

                # catch nans
                diagmat_post_cov = torch.nan_to_num(diagmat_post_cov, nan=1.0)

                # convert to required gpytorch.distribution.MultivariateNormal
                mvn = MultivariateNormal(post_mean, diagmat_post_cov)
                self.posterior_distribution = GPyTorchPosterior(mvn)

                if 'return_samples' in kwargs:
                    if kwargs['return_samples']:
                        return self.posterior_distribution, fs_sim
                return self.posterior_distribution

        # return default posterior if not constrained evaluation in kwargs
        self.posterior_distribution = super().posterior(X=X, observation_noise=observation_noise, **kwargs)
        return self.posterior_distribution

    def get_factors(self):
        if not isinstance(self.L, Tensor):
            self.L = psd_safe_cholesky(self.covar_train_train_noisy, max_tries=6)
        L = self.L

        if not isinstance(self.v1, Tensor):
            self.v1 = torch.triangular_solve(self.covar_train_ddxv, self.L, upper=False).solution
        v1 = self.v1

        if not isinstance(self.A1, Tensor):
            self.A1 = torch.triangular_solve(self.v1, self.L.T, upper=True).solution.T
        A1 = self.A1

        if not isinstance(self.B1, Tensor):
            covar_ddxv_ddxv_noisy = self.covar_ddxv_ddxv + torch.eye(
                self.covar_ddxv_ddxv.shape[-1]) * self.location_noise
            self.B1 = covar_ddxv_ddxv_noisy - v1.T @ v1
        B1 = self.B1

        if not isinstance(self.L1, Tensor):
            self.L1 = psd_safe_cholesky(B1, max_tries=6)
        L1 = self.L1

        return L, v1, A1, B1, L1

    def get_derivative_covars(self, X, training_inputs):
        batch_mode = len(X.shape) == 3
        X, virtual_obs_points = _match_batch_dims(X, self.VOPs)

        # get covariances matrices of the partial derivatives of constrained spatio dimensions
        spatio_scale = self.spatio_kernel.outputscale

        training_inputs_nobatch = training_inputs[0, ...] if batch_mode else training_inputs
        virtual_obs_points_nobatch = virtual_obs_points[0, ...] if batch_mode else virtual_obs_points

        # covar(X, X_v)
        if not isinstance(self.covar_train_ddxv, Tensor):
            covar_train_ddxv = self.spatio_kernel.base_kernel.construct_k_x1_ddx2(
                training_inputs_nobatch, virtual_obs_points_nobatch)
            if self.unconstr_dims:
                tmpr_covar_train_ddxv = self.unconstrained_kernel(training_inputs_nobatch,
                                                             virtual_obs_points_nobatch).evaluate()
                tmpr_covar_train_ddxv = torch.tile(tmpr_covar_train_ddxv, (1, len(self.constr_dims)))
                covar_train_ddxv = covar_train_ddxv * tmpr_covar_train_ddxv
            self.covar_train_ddxv = covar_train_ddxv * spatio_scale
        else:
            covar_train_ddxv = self.covar_train_ddxv
        _, covar_train_ddxv = _match_batch_dims(X, covar_train_ddxv)

        # covar(X_v, X_v)
        if not isinstance(self.covar_ddxv_ddxv, Tensor):
            covar_ddxv_ddxv = self.spatio_kernel.base_kernel.construct_k_ddx1_ddx2(
                virtual_obs_points_nobatch, virtual_obs_points_nobatch)
            if self.unconstr_dims:
                tmpr_covar_ddxv_ddxv = self.unconstrained_kernel(virtual_obs_points_nobatch,
                                                            virtual_obs_points_nobatch).evaluate()
                tmpr_covar_ddxv_ddxv = torch.tile(tmpr_covar_ddxv_ddxv,
                                                  (len(self.constr_dims), len(self.constr_dims)))
                covar_ddxv_ddxv = covar_ddxv_ddxv * tmpr_covar_ddxv_ddxv
            self.covar_ddxv_ddxv = covar_ddxv_ddxv * spatio_scale
        else:
            covar_ddxv_ddxv = self.covar_ddxv_ddxv
        _, covar_ddxv_ddxv = _match_batch_dims(X, covar_ddxv_ddxv)

        # covar(X_*, X_v)
        covar_test_ddxv = self.spatio_kernel.base_kernel.construct_k_x1_ddx2(X, virtual_obs_points)
        if self.unconstr_dims:
            tmpr_covar_test_ddxv = self.unconstrained_kernel(X, virtual_obs_points).evaluate()
            if batch_mode:
                tmpr_covar_test_ddxv = torch.tile(tmpr_covar_test_ddxv, (1, 1, len(self.constr_dims)))
            else:
                tmpr_covar_test_ddxv = torch.tile(tmpr_covar_test_ddxv, (1, len(self.constr_dims)))
            covar_test_ddxv = covar_test_ddxv * tmpr_covar_test_ddxv
        covar_test_ddxv = covar_test_ddxv * spatio_scale

        return covar_train_ddxv, covar_ddxv_ddxv, covar_test_ddxv

    def get_samples_from_prior(self, X, samples, **kwargs):

        # get joint distribution of training and testing
        full_mean, full_covar = self.get_joint_distribution(X)

        # assign mean and covariances
        test_mean = full_mean[..., self.num_train:]
        covar_test_test = full_covar[..., self.num_train:, self.num_train:].evaluate()

        if 'evaluate_constrained' not in kwargs:
            mvn = MultivariateNormal(test_mean, covar_test_test)
            return GPyTorchPosterior(mvn).sample(sample_shape=torch.Size([samples])).detach().numpy()

        else:
            if kwargs['evaluate_constrained'] is True:
                # get obseration points
                self.VOPs = kwargs['virtual_oberservation_points']
                if not isinstance(self.VOPs, Tensor):
                    raise RuntimeError("Virtual observations must be a tensor!")

                if 'nr_samples' in kwargs:
                    print('Nr. of samples is already defined in the function call.')

                _, covar_ddxv_ddxv, covar_test_ddxv = self.get_derivative_covars(X, self.optimized_VOP)

                # calculate factors (Agrell 2019), TODO: Speed up by saving v1, L1, B1
                covar_ddxv_ddxv_noisy = torch.add(covar_ddxv_ddxv,
                                                  torch.eye(covar_ddxv_ddxv.shape[-1]) * self.location_noise)

                B1 = covar_ddxv_ddxv_noisy
                B2 = covar_test_test
                B3 = covar_test_ddxv
                L1 = psd_safe_cholesky(B1)
                L1_transpose = L1.transpose(dim0=-2, dim1=-1)  # for numerical stability

                v3 = torch.triangular_solve(B3.transpose(dim0=-2, dim1=-1), L1, upper=False).solution

                A = torch.triangular_solve(v3, L1_transpose, upper=True).solution.transpose(dim0=-2, dim1=-1)

                sigma = B2 - v3.transpose(dim0=-2, dim1=-1) @ v3
                Q = psd_safe_cholesky(sigma, max_tries=6)

                cov_s = B1
                mu_s = torch.zeros(cov_s.shape[0])

                C = self.sample_from_truncated_norm(mu_s, cov_s, n_samples=samples,
                                                    sampling_method='minimax_tilting')

                # draw samples (dim(x_test) x n_samples) from multivariate standard distribution
                m = torch.distributions.MultivariateNormal(torch.zeros(Q.shape[-1]), torch.eye(Q.shape[-1]))
                U = m.rsample(sample_shape=torch.Size([C.shape[-1]])).T

                # calculate mean
                prior_test_mean = torch.tile(test_mean.unsqueeze(-1), (1, C.shape[-1]))

                # convert samples to type of the input
                C, U = _match_dtype(X, C, U)

                # approximate prior f*| C
                fs_sim = prior_test_mean + A @ C + Q @ U
                return fs_sim
            else:
                mvn = MultivariateNormal(test_mean, covar_test_test)
                return GPyTorchPosterior(mvn).sample(sample_shape=torch.Size([samples])).detach()

    def get_samples_from_last_posterior(self, n_samples):
        if isinstance(self.prev_post_samples, Tensor):
            perm = torch.randperm(self.prev_post_samples.size(1))
            idx = perm[:n_samples]
            return self.prev_post_samples[:, idx].clone().detach()
        else:
            sample = self.posterior_distribution.sample(sample_shape=torch.Size([n_samples])).detach().T
            return sample.squeeze(0)

    def reset_samples(self):
        self.C = None
        self.optimized_VOP = None

    def reset_factors(self):
        self.L = None
        self.covar_train_ddxv = None
        self.covar_ddxv_ddxv = None
        self.centered_mean = None
        self.centered_mean_nobatch = None
        self.L1 = None
        self.B1 = None
        self.v1 = None
        self.A1 = None

    def sample_from_truncated_norm(self, mu_trunc, cov_trunc, n_samples=1000, sampling_method='minimax_tilting'):
        ndim = mu_trunc.shape[0]

        # convert to numpy for sampling
        if isinstance(mu_trunc, Tensor):
            mu_trunc = mu_trunc.detach().numpy()
            cov_trunc = cov_trunc.detach().numpy()

        if sampling_method == 'minimax_tilting':
            lb = np.ones(ndim) * self.bounds[0]
            ub = np.ones(ndim) * self.bounds[1]

            t0 = time.time()
            print(f"Start minimax algorithm (Python implementation) for {n_samples} samples (D={ndim})...")
            tn = TruncatedMVN(mu_trunc, cov_trunc, lb, ub)
            samples = tn.sample(n_samples)
            print(f"Time taken for {n_samples} samples: {time.time() - t0:>4.2f}s.")
            return torch.from_numpy(samples)
        else:
            raise NotImplementedError

    @property
    def num_train(self):
        return len(self.train_inputs[0])

    def get_joint_distribution(self, X):
        # Concatenate the input to the training inputs
        train_inputs = list(self.train_inputs) if self.train_inputs is not None else []
        inputs = [X]
        full_inputs = []
        batch_shape = train_inputs[0].shape[:-2]
        for train_input, input in zip(train_inputs, inputs):
            # Make sure the batch shapes agree for training/test data
            if batch_shape != train_input.shape[:-2]:
                batch_shape = _mul_broadcast_shape(batch_shape, train_input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
            if batch_shape != input.shape[:-2]:
                batch_shape = _mul_broadcast_shape(batch_shape, input.shape[:-2])
                train_input = train_input.expand(*batch_shape, *train_input.shape[-2:])
                input = input.expand(*batch_shape, *input.shape[-2:])
            full_inputs.append(torch.cat([train_input, input], dim=-2))

        # Get the joint distribution for training/test data
        full_output = super(ExactGP, self).__call__(*full_inputs)
        if settings.debug().on():
            if not isinstance(full_output, MultivariateNormal):
                raise RuntimeError("ExactGP.forward must return a MultivariateNormal!")
        return full_output.loc, full_output.lazy_covariance_matrix