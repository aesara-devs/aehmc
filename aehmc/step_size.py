from typing import Callable, Tuple

import aesara.tensor as at
from aesara.tensor.var import TensorVariable

from aehmc import algorithms


def dual_averaging_adaptation(
    target_acceptance_rate: TensorVariable = at.as_tensor(0.8),
    gamma: float = 0.05,
    t0: int = 10,
    kappa: float = 0.75,
) -> Tuple[Callable, Callable]:
    r"""Tune the step size to achieve a desired target acceptance rate.

    Let us note :math:`\epsilon` the current step size, :math:`\alpha_t` the
    metropolis acceptance rate at time :math:`t` and :math:`\delta` the desired
    aceptance rate. We define:

    .. math:
        H_t = \delta - \alpha_t

    the error at time t. We would like to find a procedure that adapts the
    value of :math:`\epsilon` such that :math:`h(x) =\mathbb{E}\left[H_t|\epsilon\right] = 0`
    Following [1]_, the authors of [2]_ proposed the following update scheme. If
    we note :math:``x = \log \epsilon` we follow:

    .. math:
        x_{t+1} \LongLeftArrow \mu - \frac{\sqrt{t}}{\gamma} \frac{1}{t+t_0} \sum_{i=1}^t H_i
        \overline{x}_{t+1} \LongLeftArrow x_{t+1}\\, t^{-\kappa}  + \left(1-t^\kappa\right)\overline{x}_t

    :math:`\overline{x}_{t}` is guaranteed to converge to a value such that
    :math:`h(\overline{x}_t)` converges to 0, i.e. the Metropolis acceptance
    rate converges to the desired rate.

    See reference [2]_ (section 3.2.1) for a detailed discussion.

    Parameters
    ----------
    initial_log_step_size:
        Initial value of the logarithm of the step size, used as an iterate in
        the dual averaging algorithm.
    target_acceptance_rate:
        Target acceptance rate.
    gamma
        Controls the speed of convergence of the scheme. The authors of [2]_ recommend
        a value of 0.05.
    t0: float >= 0
        Free parameter that stabilizes the initial iterations of the algorithm.
        Large values may slow down convergence. Introduced in [2]_ with a default
        value of 10.
    kappa: float in ]0.5, 1]
        Controls the weights of past steps in the current update. The scheme will
        quickly forget earlier step for a small value of `kappa`. Introduced
        in [2]_, with a recommended value of .75

    Returns
    -------
    init
        A function that initializes the state of the dual averaging scheme.
    update
        A function that updates the state of the dual averaging scheme.

    References
    ----------
    .. [1]: Nesterov, Yurii. "Primal-dual subgradient methods for convex
            problems." Mathematical programming 120.1 (2009): 221-259.
    .. [2]: Hoffman, Matthew D., and Andrew Gelman. "The No-U-Turn sampler:
            adaptively setting path lengths in Hamiltonian Monte Carlo." Journal
            of Machine Learning Research 15.1 (2014): 1593-1623.

    """
    da_init, da_update = algorithms.dual_averaging(gamma, t0, kappa)

    def update(
        acceptance_probability: TensorVariable,
        step: TensorVariable,
        log_step_size: TensorVariable,
        log_step_size_avg: TensorVariable,
        gradient_avg: TensorVariable,
        mu: TensorVariable,
    ) -> Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable]:
        """Update the dual averaging adaptation state.

        Parameters
        ----------
        acceptance_probability
            The acceptance probability returned by the sampling algorithm.
        step
            The current step number.
        log_step_size
            The logarithm of the current value of the current step size.
        log_step_size_avg
            The average of the logarithm of the current step size.
        gradient_avg
            The current value of the averaged gradients.
        mu
            The points towards which iterates are shrunk.

        """
        gradient = target_acceptance_rate - acceptance_probability
        return da_update(
            gradient, step, log_step_size, log_step_size_avg, gradient_avg, mu
        )

    return da_init, update
