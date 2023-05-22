from typing import Callable, Dict, Tuple

import aesara.tensor as at
import numpy as np
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc import hmc, integrators, metrics
from aehmc.integrators import IntegratorState
from aehmc.proposals import ProposalState
from aehmc.termination import iterative_uturn
from aehmc.trajectory import Diagnostics, dynamic_integration, multiplicative_expansion

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
        state: IntegratorState,
        step_size: TensorVariable,
        inverse_mass_matrix: TensorVariable,
    ) -> Tuple[Diagnostics, Dict]:
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

        initial_state = state._replace(momentum=momentum_generator(srng))
        initial_termination_state = new_termination_state(
            initial_state.position, max_num_expansions
        )
        initial_energy = initial_state.potential_energy + kinetic_energy_fn(
            initial_state.momentum
        )
        initial_proposal = ProposalState(
            state=initial_state,
            energy=initial_energy,
            weight=at.as_tensor(0.0, dtype=np.float64),
            sum_log_p_accept=at.as_tensor(-np.inf, dtype=np.float64),
        )

        results, updates = expand(
            initial_proposal,
            initial_state,
            initial_state,
            initial_state.momentum,
            initial_termination_state,
            initial_energy,
            step_size,
        )

        # extract the last iteration from multiplicative_expansion chain diagnostics
        chain_info = Diagnostics(
            state=IntegratorState(
                position=results.diagnostics.state.position[-1],
                momentum=results.diagnostics.state.momentum[-1],
                potential_energy=results.diagnostics.state.potential_energy[-1],
                potential_energy_grad=results.diagnostics.state.potential_energy_grad[
                    -1
                ],
            ),
            acceptance_probability=results.diagnostics.acceptance_probability[-1],
            num_doublings=results.diagnostics.num_doublings[-1],
            is_turning=results.diagnostics.is_turning[-1],
            is_diverging=results.diagnostics.is_diverging[-1],
        )

        return chain_info, updates

    return step
