from typing import Callable, Tuple

import aesara
import aesara.tensor as aet
import numpy as np
from aesara.ifelse import ifelse
from aesara.scan.utils import until
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc.integrators import IntegratorStateType
from aehmc.proposals import (
    ProposalStateType,
    progressive_biased_sampling,
    progressive_uniform_sampling,
    proposal_generator,
)

__all__ = ["static_integration", "dynamic_integration", "multiplicative_expansion"]

TerminationStateType = Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable
]


# -------------------------------------------------------------------
#                       STATIC INTEGRATION
#
# This section contains algorithms that integrate the trajectory for
# a set number of integrator steps.
# -------------------------------------------------------------------


def static_integration(
    integrator: Callable,
    num_integration_steps: int,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    def integrate(
        q_init, p_init, energy_init, energy_grad_init, step_size
    ) -> IntegratorStateType:
        def one_step(q, p, potential_energy, potential_energy_grad):
            new_state = integrator(
                q, p, potential_energy, potential_energy_grad, step_size
            )
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
    srng: RandomStream,
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
    generate_proposal = proposal_generator(kinetic_energy, divergence_threshold)
    sample_proposal = progressive_uniform_sampling

    def integrate(
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
                0,
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

        integration_step = aet.as_tensor_variable(np.array(1, dtype=np.int32))
        traj, updates = aesara.scan(
            add_one_state,
            outputs_info=(
                integration_step,
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
            n_steps=max_num_steps + 1,
        )

        new_proposal = (
            (traj[1][-1], traj[2][-1], traj[3][-1], traj[4][-1]),
            traj[5][-1],
            traj[6][-1],
            traj[7][-1],
        )
        new_state = (traj[8][-1], traj[9][-1], traj[10][-1], traj[11][-1])
        subtree_momentum_sum = traj[12][-1]
        new_termination_state = (traj[13][-1], traj[14][-1], traj[15][-1], traj[16][-1])
        is_diverging = traj[-2][-1]
        has_terminated = traj[-1][-1]

        return (
            new_proposal,
            new_state,
            subtree_momentum_sum,
            new_termination_state,
            is_diverging,
            has_terminated,
        ), updates

    return integrate


def multiplicative_expansion(
    srng: RandomStream,
    trajectory_integrator: Callable,
    uturn_check_fn: Callable,
    step_size: TensorVariable,
    max_num_expansions: TensorVariable,
):
    """Sample a trajectory and update the proposal sequentially
    until the termination criterion is met.

    The trajectory is sampled with the following procedure:
    1. Pick a direction at random;
    2. Integrate `num_step` steps in this direction;
    3. If the integration has stopped prematurely, do not update the proposal;
    4. Else if the trajectory is performing a U-turn, return current proposal;
    5. Else update proposal, `num_steps = num_steps ** rate` and repeat from (1).

    Parameters
    ----------
    trajectory_integrator
        A function that runs the symplectic integrators and returns a new proposal
        and the integrated trajectory.
    uturn_check_fn
        Function used to check the U-Turn criterion.
    step_size
        The step size used by the symplectic integrator.
    max_num_expansions
        The maximum number of trajectory expansions until the proposal is
        returned.

    """
    proposal_sampler = progressive_biased_sampling

    def expand(
        proposal,
        left_state,
        right_state,
        momentum_sum,
        termination_state,
        initial_energy,
    ):
        def expand_once(
            step,
            q_proposal,  # proposal
            p_proposal,
            potential_energy_proposal,
            potential_energy_grad_proposal,
            energy_proposal,
            weight,
            sum_log_p_accept,
            q_left,  # trajectory
            p_left,
            potential_energy_left,
            potential_energy_grad_left,
            q_right,
            p_right,
            potential_energy_right,
            potential_energy_grad_right,
            momentum_sum,
            momentum_ckpts,  # termination_state
            momentum_sum_ckpts,
            idx_min,
            idx_max,
            is_diverging,
            has_terminated,
            do_stop_expanding,
        ):
            """Expand the current trajectory.

            At each step we draw a direction at random, build a subtrajectory starting
            from the leftmost or rightmost point of the current trajectory that is
            twice as long as the current trajectory.

            Once that is done, possibly update the current proposal with that of
            the subtrajectory.

            """

            left_state = (
                q_left,
                p_left,
                potential_energy_left,
                potential_energy_grad_left,
            )
            right_state = (
                q_right,
                p_right,
                potential_energy_right,
                potential_energy_grad_right,
            )
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
            termination_state = (
                momentum_ckpts,
                momentum_sum_ckpts,
                idx_min,
                idx_max,
            )

            do_go_right = srng.bernoulli(0.5)
            direction = aet.where(do_go_right, 1.0, -1.0)
            start_state = where_state(do_go_right, right_state, left_state)

            (
                new_proposal,
                new_state,
                subtree_momentum_sum,
                new_termination_state,
                is_diverging,
                has_subtree_terminated,
            ), inner_updates = trajectory_integrator(
                start_state,
                direction,
                termination_state,
                2 ** step,
                step_size,
                initial_energy,
            )
            for key, value in inner_updates.items():
                key.default_update = value

            new_momentum_sum = momentum_sum + subtree_momentum_sum

            # The trajectory integrator always integrates forward in time; we
            # thus need to switch the states if the other direction was picked.
            new_left_state = where_state(do_go_right, left_state, new_state)
            new_right_state = where_state(do_go_right, new_state, right_state)

            # Update the proposal If the termination criterion is reached in
            # the subtree or if a divergence occurs we reject this subtree's
            # proposal. We nevertheless update the sum of the logarithm of the
            # acceptance probabilities to serve as an estimate for dual
            # averaging.
            updated_sum_log_p_accept = _logaddexp(proposal[3], new_proposal[3])
            updated_proposal = (
                proposal[0],
                proposal[1],
                proposal[2],
                updated_sum_log_p_accept,
            )

            sampled_proposal = where_proposal(
                is_diverging,
                updated_proposal,
                proposal_sampler(srng, proposal, new_proposal),
            )

            # check if the trajectory is turning
            is_turning = uturn_check_fn(
                new_left_state[1], new_right_state[1], new_momentum_sum
            )

            do_stop_expanding = is_diverging | is_turning
            new_proposal = proposal_sampler(srng, proposal, new_proposal)

            return (
                step + 1,
                *sampled_proposal[0],
                sampled_proposal[1],
                sampled_proposal[2],
                sampled_proposal[3],
                *new_left_state,
                *new_right_state,
                new_momentum_sum,
                *new_termination_state,
                is_diverging,
                is_turning,
                has_subtree_terminated,
            ), until(do_stop_expanding)

        expansion_step = aet.as_tensor_variable(np.array(0, dtype=np.int32))
        results, updates = aesara.scan(
            expand_once,
            outputs_info=(
                expansion_step,
                *proposal[0],
                proposal[1],
                proposal[2],
                proposal[3],
                *left_state,
                *right_state,
                momentum_sum,
                *termination_state,
                np.array(False),
                np.array(False),
                np.array(False),
            ),
            n_steps=max_num_expansions,
        )
        for key, value in updates.items():
            key.default_update = value

        return results, updates

    return expand


def where_state(
    do_pick_left: bool,
    left_state: IntegratorStateType,
    right_state: IntegratorStateType,
) -> IntegratorStateType:
    """Represents a switch between two states depending on a condition."""
    q_left, p_left, potential_energy_left, potential_energy_grad_left = left_state
    q_right, p_right, potential_energy_right, potential_energy_grad_right = right_state

    q = ifelse(do_pick_left, q_left, q_right)
    p = ifelse(do_pick_left, p_left, p_right)
    potential_energy = aet.where(
        do_pick_left, potential_energy_left, potential_energy_right
    )
    potential_energy_grad = ifelse(
        do_pick_left, potential_energy_grad_left, potential_energy_grad_right
    )

    return (q, p, potential_energy, potential_energy_grad)


def where_proposal(
    do_pick_left: bool,
    left_proposal: ProposalStateType,
    right_proposal: ProposalStateType,
) -> ProposalStateType:
    """Represents a switch between two proposals depending on a condition."""
    left_state, left_weight, left_energy, left_log_sum_p_accept = left_proposal
    right_state, right_weight, right_energy, right_log_sum_p_accept = right_proposal

    state = where_state(do_pick_left, left_state, right_state)
    energy = aet.where(do_pick_left, left_energy, right_energy)
    weight = aet.where(do_pick_left, left_weight, right_weight)
    log_sum_p_accept = ifelse(
        do_pick_left, left_log_sum_p_accept, right_log_sum_p_accept
    )

    return (state, energy, weight, log_sum_p_accept)


def _logaddexp(a, b):
    diff = b - a
    return aet.switch(
        diff > 0, b + aet.log1p(aet.exp(-diff)), a + aet.log1p(aet.exp(-diff))
    )
