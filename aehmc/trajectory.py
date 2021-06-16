from typing import Callable, Tuple

import aesara
import numpy as np
from aesara.scan.utils import until
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc.integrators import IntegratorStateType
from aehmc.proposals import (
    ProposalStateType,
    progressive_uniform_sampling,
    proposal_generator,
)

TerminationStateType = Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable
]

TrajectoryType = Tuple[
    IntegratorStateType, IntegratorStateType, TensorVariable, TensorVariable
]


def append_to_trajectory(trajectory, new_state):
    """Append a state to the right of a trajectory."""
    return (trajectory[0], new_state, trajectory[2] + new_state[1], trajectory[3] + 1)


# -------------------------------------------------------------------
#                       STATIC INTEGRATION
#
# This section contains algorithms that integrate the trajectory for
# a set number of integrator steps.
# -------------------------------------------------------------------


def static_integration(
    integrator: Callable,
    step_size: float,
    num_integration_steps: int,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    directed_step_size = direction * step_size

    def integrate(q_init, p_init, energy_init, energy_grad_init) -> IntegratorStateType:
        def one_step(q, p, energy, energy_grad):
            new_state = integrator(q, p, energy, energy_grad, directed_step_size)
            return new_state

        [q, p, energy, energy_grad], _ = aesara.scan(
            fn=one_step,
            outputs_info=[
                {"initial": q_init},
                {"initial": p_init},
                {"initial": energy_init},
                {"initial": energy_grad_init},
            ],
            n_steps=num_integration_steps,
        )

        return q[-1], p[-1], energy[-1], energy_grad[-1]

    return integrate


# -------------------------------------------------------------------
#                       DYNAMIC INTEGRATION
#
# This section contains algorithms that determine the number of
# integrator steps dynamically using a termination criterion that
# is updated at every step.
# -------------------------------------------------------------------


def dynamic_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    update_termination_state: Callable,
    is_criterion_met: Callable,
    divergence_threshold: TensorVariable,
):
    """Integrate a trajectory and update the proposal sequentially in one direction
    until the termination criterion is met.

    Parameters
    ----------
    integrator
        The symplectic integrator used to integrate the hamiltonian trajectory.
    kinetic_energy
        Function to compute the current value of the kinetic energy.
    update_termination_state
        Updates the state of the termination mechanism.
    is_criterion_met
        Determines whether the termination criterion has been met.
    divergence_threshold
        Value of the difference of energy between two consecutive states above which we say a transition is divergent.

    """
    _, generate_proposal = proposal_generator(kinetic_energy, divergence_threshold)
    sample_proposal = progressive_uniform_sampling

    def integrate(
        srng: RandomStream,
        previous_last_state: IntegratorStateType,
        direction: TensorVariable,
        termination_state: TerminationStateType,
        max_num_steps: TensorVariable,
        step_size: TensorVariable,
        initial_energy: TensorVariable,
    ):
        """Integrate the trajectory starting from `initial_state` and update
        the proposal sequentially until the termination criterion is met.

        Parameters
        ----------
        rng_key
            Key used by JAX's random number generator.
        previous_last_state
            The last state of the previously integrated trajectory.
        direction int in {-1, 1}
            The direction in which to expand the trajectory.
        termination_state
            The state that keeps track of the information needed for the termination criterion.
        max_num_steps
            The maximum number of integration steps. The expansion will stop
            when this number is reached if the termination criterion has not
            been met.
        step_size
            The step size of the symplectic integrator.
        initial_energy
            Initial energy H0 of the HMC step (not to confused with the initial energy of the subtree)

        """

        def take_first_step(
            previous_last_state: IntegratorStateType,
            termination_state: TerminationStateType,
        ) -> Tuple[IntegratorStateType, ProposalStateType, TerminationStateType]:
            """The first state of the new trajectory is obtained by integrating
            once starting from the last state of the previous trajectory.

            """
            initial_state = integrator(*previous_last_state, direction * step_size)
            initial_proposal, _ = generate_proposal(initial_energy, initial_state)
            initial_termination_state = update_termination_state(
                termination_state,
                initial_state[1],
                initial_state[1],
                1,
            )
            return (
                initial_state,
                initial_proposal,
                initial_termination_state,
            )

        def add_one_state(
            step,
            q_proposal,
            p_proposal,
            potential_energy_proposal,
            potential_energy_grad_proposal,
            energy_proposal,
            weight,
            sum_log_p_accept,
            q_last,
            p_last,
            potential_energy_last,
            potential_energy_grad_last,
            momentum_sum: TensorVariable,
            momentum_ckpts,
            momentum_sum_ckpts,
            idx_min,
            idx_max,
            is_diverging,
            has_terminated,
        ):
            last_state = (
                q_last,
                p_last,
                potential_energy_last,
                potential_energy_grad_last,
            )
            termination_state = (momentum_ckpts, momentum_sum_ckpts, idx_min, idx_max)
            proposal = (
                (
                    q_proposal,
                    p_proposal,
                    potential_energy_proposal,
                    potential_energy_grad_proposal,
                ),
                energy_proposal,
                weight,
                sum_log_p_accept,
            )

            new_state = integrator(*last_state, direction * step_size)
            new_proposal, is_diverging = generate_proposal(initial_energy, new_state)
            sampled_proposal = sample_proposal(srng, proposal, new_proposal)

            momentum = new_state[1]
            momentum_sum = momentum_sum + momentum
            new_termination_state = update_termination_state(
                termination_state, momentum_sum, momentum, step
            )
            has_terminated = is_criterion_met(
                new_termination_state, momentum_sum, momentum
            )

            do_stop_integrating = is_diverging | has_terminated

            return (
                step + 1,
                *sampled_proposal[0],
                sampled_proposal[1],
                sampled_proposal[2],
                sampled_proposal[3],
                *new_state,
                momentum_sum,
                *new_termination_state,
                is_diverging,
                has_terminated,
            ), until(do_stop_integrating)

        initial_state, initial_proposal, initial_termination_state = take_first_step(
            previous_last_state, termination_state
        )

        traj, updates = aesara.scan(
            add_one_state,
            outputs_info=(
                2,
                *initial_proposal[0],
                initial_proposal[1],
                initial_proposal[2],
                initial_proposal[3],
                *initial_state,
                initial_state[1],
                *termination_state,
                np.array(False),
                np.array(False),
            ),
            n_steps=max_num_steps,
        )

        is_diverging = traj[-2][-1]
        has_terminated = traj[-1][-1]

        return (is_diverging, has_terminated), updates

    return integrate


def multiplicative_expansion():
    raise NotImplementedError
