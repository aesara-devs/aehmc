from typing import Callable, Dict, Tuple

import aesara
import aesara.tensor as at
import numpy as np
from aesara.ifelse import ifelse
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

import aehmc.integrators as integrators
import aehmc.metrics as metrics
import aehmc.trajectory as trajectory


def new_state(
    q: TensorVariable, logprob_fn: Callable
) -> Tuple[TensorVariable, TensorVariable, TensorVariable]:
    """Create a new HMC state from a position.

    Parameters
    ----------
    q
        The chain's position.
    logprob_fn
        The function that computes the value of the log-probability density
        function at any position.

    Returns
    -------
    A new HMC state, i.e. a tuple with the position, current value of the
    potential energy and gradient of the potential energy.

    """
    potential_energy = -logprob_fn(q)
    potential_energy_grad = aesara.grad(potential_energy, wrt=q)
    return q, potential_energy, potential_energy_grad


def new_kernel(
    srng: RandomStream,
    logprob_fn: Callable,
    divergence_threshold: int = 1000,
) -> Callable:
    """Build a HMC kernel.

    Parameters
    ----------
    srng
        A RandomStream object that tracks the changes in a shared random state.
    logprob_fn
        A function that returns the value of the log-probability density
        function of a chain at a given position.
    divergence_threshold
        The difference in energy above which we say the transition is
        divergent.

    Returns
    -------
    A kernel that takes the current state of the chain and that returns a new
    state.

    References
    ----------
    .. [0]: Neal, Radford M. "MCMC using Hamiltonian dynamics." Handbook of markov
            chain monte carlo 2.11 (2011): 2.


    """

    def potential_fn(x):
        return -logprob_fn(x)

    def step(
        q: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
        step_size: TensorVariable,
        inverse_mass_matrix: TensorVariable,
        num_integration_steps: int,
    ) -> Tuple[
        Tuple[TensorVariable, TensorVariable, TensorVariable, TensorVariable, bool],
        Dict,
    ]:
        """Perform a single step of the HMC algorithm.

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
        num_integration_steps
            The number of times we run the integrator at each step.

        Returns
        -------
        A tuple that contains on the one hand: the new position, value of the
        potential energy, gradient of the potential energy and acceptance
        propbability. On the other hand a dictionaruy that contains the update
        rules for the shared variables updated in the scan operator.

        """

        momentum_generator, kinetic_energy_fn, _ = metrics.gaussian_metric(
            inverse_mass_matrix
        )
        symplectic_integrator = integrators.velocity_verlet(
            potential_fn, kinetic_energy_fn
        )
        proposal_generator = hmc_proposal(
            symplectic_integrator,
            kinetic_energy_fn,
            num_integration_steps,
            divergence_threshold,
        )

        p = momentum_generator(srng)
        (
            q_new,
            _,
            potential_energy_new,
            potential_energy_grad_new,
            acceptance_probability,
            is_divergent,
        ), updates = proposal_generator(
            srng, q, p, potential_energy, potential_energy_grad, step_size
        )

        return (
            q_new,
            potential_energy_new,
            potential_energy_grad_new,
            acceptance_probability,
            is_divergent,
        ), updates

    return step


def hmc_proposal(
    integrator: Callable,
    kinetic_energy: Callable[[TensorVariable], TensorVariable],
    num_integration_steps: TensorVariable,
    divergence_threshold: int,
) -> Callable:
    """Builds a function that returns a HMC proposal.

    Parameters
    --------
    integrator
        The symplectic integrator used to integrate the hamiltonian dynamics.
    kinetic_energy
        The function used to compute the kinetic energy.
    num_integration_steps
        The number of times we need to run the integrator every time the
        returned function is called.
    divergence_threshold
        The difference in energy above which we say the transition is
        divergent.

    Returns
    -------
    A function that generates a new state for the chain.

    """
    integrate = trajectory.static_integration(integrator, num_integration_steps)

    def propose(
        srng: RandomStream,
        q: TensorVariable,
        p: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
        step_size: TensorVariable,
    ) -> Tuple[
        Tuple[
            TensorVariable,
            TensorVariable,
            TensorVariable,
            TensorVariable,
            TensorVariable,
            TensorVariable,
        ],
        Dict,
    ]:
        """Use the HMC algorithm to propose a new state.

        Parameters
        ----------
        srng
            A RandomStream object that tracks the changes in a shared random state.
        q
            The initial position.
        potential_energy
            The initial value of the potential energy.
        potential_energy_grad
            The initial value of the gradient of the potential energy wrt the position.
        step_size
            The step size used in the symplectic integrator

        Returns
        -------
        A tuple that contains on the one hand: the new position, value of the
        potential energy, gradient of the potential energy and acceptance
        probability. On the other hand a dictionary that contains the update
        rules for the shared variables updated in the scan operator.

        """

        (
            new_q,
            new_p,
            new_potential_energy,
            new_potential_energy_grad,
        ), updates = integrate(q, p, potential_energy, potential_energy_grad, step_size)

        # flip the momentum to keep detailed balance
        flipped_p = -1.0 * new_p

        # compute transition-related quantities
        energy = potential_energy + kinetic_energy(p)
        new_energy = new_potential_energy + kinetic_energy(flipped_p)
        delta_energy = energy - new_energy
        delta_energy = at.where(at.isnan(delta_energy), -np.inf, delta_energy)
        is_transition_divergent = at.abs(delta_energy) > divergence_threshold

        p_accept = at.clip(at.exp(delta_energy), 0, 1.0)
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

        return (
            final_q,
            final_p,
            final_potential_energy,
            final_potential_energy_grad,
            p_accept,
            is_transition_divergent,
        ), updates

    return propose
