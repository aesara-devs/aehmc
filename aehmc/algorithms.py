from typing import Callable, Tuple

import aesara.tensor as at
import numpy as np
from aesara import config
from aesara.tensor.var import TensorVariable


def dual_averaging(
    gamma: float = 0.05, t0: int = 10, kappa: float = 0.75
) -> Tuple[Callable, Callable]:
    """Dual averaging algorithm.

    Dual averaging is an algorithm for stochastic optimization that was
    originally proposed by Nesterov in [1]_.

    The update scheme we implement here is more elaborate than the one
    described in [1]_. We follow [2]_ and add the parameters `t_0` and `kappa`
    which respectively improves the stability of computations in early
    iterations and set how fast the algorithm should forget past iterates.

    The default values for the parameters are taken from the Stan implementation [3]_.

    Parameters
    ----------
    gamma
        Controls the amount of shrinkage towards mu.
    t0
        Improves the stability of computations early on.
    kappa
        Controls how fast the algorithm should forget past iterates.

    References
    ----------
    .. [1]: Nesterov, Y. (2009). Primal-dual subgradient methods for convex
            problems. Mathematical programming, 120(1), 221-259.

    .. [2]: Hoffman, M. D., & Gelman, A. (2014). The No-U-Turn sampler:
            adaptively setting path lengths in Hamiltonian Monte Carlo. J. Mach. Learn.
            Res., 15(1), 1593-1623.

    .. [3]: Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B.,
            Betancourt, M., ... & Riddell, A. (2017). Stan: A probabilistic programming
            language. Journal of statistical software, 76(1), 1-32.

    """

    def init(
        mu: TensorVariable,
    ) -> Tuple[
        TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
    ]:
        """
        Parameters
        ----------
        mu
            Chosen points towards which the successive iterates are shrunk.

        """
        step = at.as_tensor(1, "step", dtype=np.int32)
        gradient_avg = at.as_tensor(0, "gradient_avg", dtype=config.floatX)
        x_t = at.as_tensor(0.0, "x_t", dtype=config.floatX)
        x_avg = at.as_tensor(0.0, "x_avg", dtype=config.floatX)
        return step, x_t, x_avg, gradient_avg, mu

    def update(
        gradient: TensorVariable,
        step: TensorVariable,
        x: TensorVariable,
        x_avg: TensorVariable,
        gradient_avg: TensorVariable,
        mu: TensorVariable,
    ) -> Tuple[
        TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable
    ]:
        """Update the state of the Dual Averaging algorithm.

        Parameters
        ----------
        gradient
            The current value of the stochastic gradient. Replaced by a
            statistic to optimize in the case of MCMC adaptation.
        step
            The number of the current step in the optimization process.
        x
            The current value of the iterate.
        x_avg
            The current value of the averaged iterates.
        gradient_avg
            The current value of the averaged gradients.

        Returns
        -------
        Updated values for the step number, iterate, averaged iterates and
        averaged gradients.

        """

        eta = 1.0 / (step + t0)
        new_gradient_avg = (1.0 - eta) * gradient_avg + eta * gradient

        new_x = mu - (at.sqrt(step) / gamma) * new_gradient_avg

        x_eta = step ** (-kappa)
        new_x_avg = x_eta * x + (1.0 - x_eta) * x_avg

        return (
            (step + 1).astype(np.int32),
            new_x.astype(config.floatX),
            new_x_avg.astype(config.floatX),
            new_gradient_avg.astype(config.floatX),
            mu,
        )

    return init, update


def welford_covariance(compute_covariance: bool) -> Tuple[Callable, Callable, Callable]:
    """Welford's online estimator of variance/covariance.

    It is possible to compute the variance of a population of values in an
    on-line fashion to avoid storing intermediate results. The naive recurrence
    relations between the sample mean and variance at a step and the next are
    however not numerically stable.

    Welford's algorithm uses the sum of square of differences
    :math:`M_{2,n} = \\sum_{i=1}^n \\left(x_i-\\overline{x_n}\right)^2`
    for updating where :math:`x_n` is the current mean and the following
    recurrence relationships

    Parameters
    ----------
    compute_covariance
        When True the algorithm returns a covariance matrix, otherwise returns
        a variance vector.

    """

    def init(n_dims: int) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        """Initialize the variance estimation.

        Parameters
        ----------
        n_dims: int
            The number of dimensions of the problem.

        """
        sample_size = at.as_tensor(0, dtype=np.int32)

        if n_dims == 0:
            return (
                at.as_tensor(0.0, dtype=config.floatX),
                at.as_tensor(0.0, dtype=config.floatX),
                sample_size,
            )

        mean = at.zeros((n_dims,), dtype=config.floatX)
        if compute_covariance:
            m2 = at.zeros((n_dims, n_dims), dtype=config.floatX)
        else:
            m2 = at.zeros((n_dims,), dtype=config.floatX)

        return mean, m2, sample_size

    def update(
        value: TensorVariable,
        mean: TensorVariable,
        m2: TensorVariable,
        sample_size: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
        """Update the averages and M2 matrix using the new value.

        Parameters
        ----------
        value: Array, shape (1,)
            The new sample (typically position of the chain) used to update m2
        mean
            The running average along each dimension
        m2
            The running value of the unnormalized variance/covariance
        sample_size
            The number of points that have currently been used to compute `mean` and `m2`.

        """
        sample_size = sample_size + 1

        delta = value - mean
        mean = mean + delta / sample_size
        updated_delta = value - mean
        if compute_covariance and mean.ndim > 0:
            m2 = m2 + at.outer(updated_delta, delta)
        else:
            m2 = m2 + updated_delta * delta

        return mean, m2, sample_size

    def final(m2: TensorVariable, sample_size: TensorVariable) -> TensorVariable:
        """Compute the covariance"""
        variance_or_covariance = m2 / (sample_size - 1)
        return variance_or_covariance

    return init, update, final
