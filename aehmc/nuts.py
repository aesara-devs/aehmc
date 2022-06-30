from typing import Callable

import aesara.tensor as at
import numpy as np
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc import hmc, integrators, metrics
from aehmc.termination import iterative_uturn
from aehmc.trajectory import dynamic_integration, multiplicative_expansion

new_state = hmc.new_state


def new_kernel(
    srng: RandomStream,
    logprob_fn: Callable[[TensorVariable], TensorVariable],
    max_num_expansions: int = 10,
    divergence_threshold: int = 1000,
) -> Callable:
    """Build an iterative NUTS kernel.

    Parameters
    ----------
    srng
        A RandomStream object that tracks the changes in a shared random state.
    logprob_fn
        A function that returns the value of the log-probability density
        function of a chain at a given position.
    max_num_expansions
        The maximum number of times we double the length of the trajectory.
        Known as the maximum tree depth in most implementations.
    divergence_threshold
        The difference in energy above which we say the transition is
        divergent.

    Returns
    -------
    A function which, given a chain state, returns a new chain state.

    References
    ----------
    .. [0]: Phan, Du, Neeraj Pradhan, and Martin Jankowiak. "Composable effects
            for flexible and accelerated probabilistic programming in NumPyro." arXiv
            preprint arXiv:1912.11554 (2019).
    .. [1]: Lao, Junpeng, et al. "tfp. mcmc: Modern markov chain monte carlo
            tools built for modern hardware." arXiv preprint arXiv:2002.01184 (2020).

    """

    def potential_fn(x):
        return -logprob_fn(x)

    def step(
        q: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
        step_size: TensorVariable,
        inverse_mass_matrix: TensorVariable,
    ):
        """Use the NUTS algorithm to propose a new state.

        Parameters
        ----------
        q
            The initial position.
        potential_energy
            The initial value of the potential energy.
        potential_energy_grad
            The initial value of the gradient of the potential energy wrt the position.
        step_size
            The step size used in the symplectic integrator
        inverse_mass_matrix
            One or two-dimensional array used as the inverse mass matrix that
            defines the euclidean metric.

        Returns
        -------
        A tuple that contains on the one hand: the new position, value of the
        potential energy, gradient of the potential energy, the acceptance
        probability, the number of times the trajectory expanded, whether the
        integration diverged, whether the trajectory turned on itself. On the
        other hand a dictionary that contains the update rules for the shared
        variables updated in the scan operator.

        """
        momentum_generator, kinetic_energy_fn, uturn_check_fn = metrics.gaussian_metric(
            inverse_mass_matrix
        )
        symplectic_integrator = integrators.velocity_verlet(
            potential_fn, kinetic_energy_fn
        )
        (
            new_termination_state,
            update_termination_state,
            is_criterion_met,
        ) = iterative_uturn(uturn_check_fn)
        trajectory_integrator = dynamic_integration(
            srng,
            symplectic_integrator,
            kinetic_energy_fn,
            update_termination_state,
            is_criterion_met,
            divergence_threshold,
        )
        expand = multiplicative_expansion(
            srng,
            trajectory_integrator,
            uturn_check_fn,
            max_num_expansions,
        )

        p = momentum_generator(srng)
        initial_state = (q, p, potential_energy, potential_energy_grad)
        initial_termination_state = new_termination_state(q, max_num_expansions)
        initial_energy = potential_energy + kinetic_energy_fn(p)
        initial_proposal = (
            initial_state,
            initial_energy,
            at.as_tensor(0.0, dtype=np.float64),
            at.as_tensor(-np.inf, dtype=np.float64),
        )
        result, updates = expand(
            initial_proposal,
            initial_state,
            initial_state,
            p,
            initial_termination_state,
            initial_energy,
            step_size,
        )

        # New MCMC proposal
        q_new = result[0][-1]
        potential_energy_new = result[2][-1]
        potential_energy_grad_new = result[3][-1]

        # Diagnostics
        is_turning = result[-1][-1]
        is_diverging = result[-2][-1]
        num_doublings = result[-3][-1]
        acceptance_probability = result[-4][-1]

        return (
            q_new,
            potential_energy_new,
            potential_energy_grad_new,
            acceptance_probability,
            num_doublings,
            is_turning,
            is_diverging,
        ), updates

    return step
