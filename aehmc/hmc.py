from typing import Callable

import aesara
import aesara.tensor as aet
import numpy as np
from aesara.ifelse import ifelse
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

import aehmc.integrators as integrators
import aehmc.metrics as metrics
import aehmc.trajectory as trajectory


def new_state(q: TensorVariable, logprob_fn: Callable):
    potential_energy = -logprob_fn(q)
    potential_energy_grad = aesara.grad(potential_energy, wrt=q)
    return q, potential_energy, potential_energy_grad


def kernel(
    srng: RandomStream,
    logprob_fn: TensorVariable,
    step_size: TensorVariable,
    inverse_mass_matrix: TensorVariable,
    num_integration_steps: TensorVariable,
    divergence_threshold: int = 1000,
):
    """Build a HMC kernel.

    Parameters
    ----------
    srng
        RandomStream object.
    logprob_fn
        A function that returns the value of the log-probability density
        function of a chain at a given position.
    step_size
        The step size used in the symplectic integrator
    inverse_mass_matrix
        One or two-dimensional array used as the inverse mass matrix that
        defines the euclidean metric.
    num_integration_steps
        The number of times we apply the symplectic integrator to integrate the trajectory.
    divergence_threshold
        The difference in energy above which we say the transition is
        divergent.

    Returns
    -------
    A kernel that takes the current state of the chain and that returns a new
    state.


    """

    def potential_fn(x):
        return -logprob_fn(x)

    momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_metric(
        inverse_mass_matrix
    )
    symplectic_integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
    proposal_generator = hmc_proposal(
        symplectic_integrator,
        kinetic_energy_fn,
        step_size,
        num_integration_steps,
        divergence_threshold,
    )

    def step(
        q: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
    ):
        """Perform a single step of the HMC algorithm."""
        p = momentum_generator(srng)
        (
            q_new,
            p_new,
            potential_energy_new,
            potential_energy_grad_new,
        ) = proposal_generator(srng, q, p, potential_energy, potential_energy_grad)
        return q_new, potential_energy_new, potential_energy_grad_new

    return step


def hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable[[TensorVariable], TensorVariable],
    step_size: TensorVariable,
    num_integration_steps: TensorVariable,
    divergence_threshold: int,
):
    """Builds a function that returns a HMC proposal."""

    integrate = trajectory.static_integration(
        integrator, step_size, num_integration_steps
    )

    def propose(
        srng: RandomStream,
        q: TensorVariable,
        p: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
    ):
        """Use the HMC algorithm to propose a new state."""

        new_q, new_p, new_potential_energy, new_potential_energy_grad = integrate(
            q, p, potential_energy, potential_energy_grad
        )

        # flip the momentum to keep detailed balance
        flipped_p = -1.0 * new_p

        # compute transition-related quantities
        energy = potential_energy + kinetic_energy(p)
        new_energy = new_potential_energy + kinetic_energy(flipped_p)
        delta_energy = energy - new_energy
        delta_energy = aet.where(aet.isnan(delta_energy), -np.inf, delta_energy)
        # is_transition_divergence = aet.abs(delta_energy) > divergence_threshold

        p_accept = aet.clip(aet.exp(delta_energy), 0, 1.0)
        do_accept = srng.bernoulli(p_accept)
        (
            final_q,
            final_p,
            final_potential_energy,
            final_potential_energy_grad,
        ) = ifelse(
            do_accept,
            (new_q, flipped_p, new_potential_energy, new_potential_energy_grad),
            (q, p, potential_energy, potential_energy_grad),
        )

        return final_q, final_p, final_potential_energy, final_potential_energy_grad

    return propose
