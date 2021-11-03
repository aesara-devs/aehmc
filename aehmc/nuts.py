from typing import Callable

import aesara.tensor as aet
import numpy as np
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc import hmc, integrators, metrics
from aehmc.termination import iterative_uturn
from aehmc.trajectory import dynamic_integration, multiplicative_expansion

new_state = hmc.new_state


def kernel(
    srng: RandomStream,
    logprob_fn: Callable[[TensorVariable], TensorVariable],
    inverse_mass_matrix: TensorVariable,
    max_num_expansions: int = aet.as_tensor(10),
    divergence_threshold: int = 1000,
) -> Callable:
    """Build an iterative NUTS kernel.


    Parameters
    ----------
    logprob_fn
        A function that returns the value of the log-probability density
        function of a chain at a given position.
    step_size
        The step size used in the symplectic integrator
    inverse_mass_matrix
        One or two-dimensional array used as the inverse mass matrix that
        defines the euclidean metric.
    max_num_expansions
        The maximum number of times we double the length of the trajectory.
        Known as the maximum tree depth in most implementations.
    divergence_threshold
        The difference in energy above which we say the transition is
        divergent.

    Return
    ------

    A function which, given a chain state, returns a new chain state.

    """

    def potential_fn(x):
        return -logprob_fn(x)

    momentum_generator, kinetic_ernergy_fn, uturn_check_fn = metrics.gaussian_metric(
        inverse_mass_matrix
    )
    symplectic_integrator = integrators.velocity_verlet(
        potential_fn, kinetic_ernergy_fn
    )
    new_termination_state, update_termination_state, is_criterion_met = iterative_uturn(
        uturn_check_fn
    )
    trajectory_integrator = dynamic_integration(
        srng,
        symplectic_integrator,
        kinetic_ernergy_fn,
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

    def step(
        q: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
        step_size: TensorVariable,
    ):
        """Move the chain by one step.

        Parameters
        ----------
        logprob_fn
            A function that returns the value of the log-probability density
            function of a chain at a given position.
        step_size
            The step size used in the symplectic integrator

        """
        p = momentum_generator(srng)
        initial_state = (q, p, potential_energy, potential_energy_grad)
        initial_termination_state = new_termination_state(q, max_num_expansions)
        initial_energy = potential_energy + kinetic_ernergy_fn(p)
        initial_proposal = (
            initial_state,
            initial_energy,
            aet.as_tensor(0.0, dtype="float64"),
            aet.as_tensor([-np.inf], dtype="float64"),
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
        for key, value in updates.items():
            key.default_update = value

        q_new = result[1][-1]
        potential_energy_new = result[3][-1]
        potential_energy_grad_new = result[4][-1]
        is_diverging = result[-3][-1]
        has_subtree_terminated = result[-1][-1]
        is_turning = result[-2][-1]
        num_steps = result[0][-1]

        return (
            q_new,
            potential_energy_new,
            potential_energy_grad_new,
            num_steps,
            is_turning,
            is_diverging,
            has_subtree_terminated,
        ), updates

    return step
