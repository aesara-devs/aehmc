from typing import Callable, Dict, Tuple

import aesara
import aesara.tensor as at
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
    """Build a function that generates fixed-length trajectories.

    Parameters
    ----------
    integrator
       Function that performs one integration step.
    num_integration_steps
        The number of times we need to run the integrator every time the
        returned function is called.

    Returns
    -------
    A function that integrates the hamiltonian dynamics a
    `num_integration_steps` times.

    """

    def integrate(
        q_init: TensorVariable,
        p_init: TensorVariable,
        energy_init: TensorVariable,
        energy_grad_init: TensorVariable,
        step_size: TensorVariable,
    ) -> Tuple[IntegratorStateType, Dict]:
        """Generate a trajectory by integrating several times in one direction.

        Parameters
        ----------
        q_init
            The initial position.
        p_init
            The initial value of the momentum.
        energy_init
            The initial value of the potential energy.
        energy_grad_init
            The initial value of the gradient of the potential energy wrt the position.
        step_size
            The size of each step taken by the integrator.

        Returns
        -------
        A tuple with the last position, value of the momentum, potential energy,
        gradient of the potential energy wrt the position in a tuple as well as
        a dictionary that contains the update rules for all the shared variables
        updated in `scan`.

        """

        def one_step(q, p, potential_energy, potential_energy_grad):
            new_state = integrator(
                q, p, potential_energy, potential_energy_grad, step_size
            )
            return new_state

        [q, p, energy, energy_grad], updates = aesara.scan(
            fn=one_step,
            outputs_info=[
                {"initial": q_init},
                {"initial": p_init},
                {"initial": energy_init},
                {"initial": energy_grad_init},
            ],
            n_steps=num_integration_steps,
        )

        return (q[-1], p[-1], energy[-1], energy_grad[-1]), updates

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
    srng
        A RandomStream object that tracks the changes in a shared random state.
    integrator
        The symplectic integrator used to integrate the hamiltonian dynamics.
    kinetic_energy
        Function to compute the current value of the kinetic energy.
    update_termination_state
        Updates the state of the termination mechanism.
    is_criterion_met
        Determines whether the termination criterion has been met.
    divergence_threshold
        Value of the difference of energy between two consecutive states above which we say a transition is divergent.

    Returns
    -------
    A function that integrates the trajectory in one direction and updates a
    proposal until the termination criterion is met.

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

        Returns
        -------
        A tuple with on the one hand: a new proposal (sampled from the states
        traversed while building the trajectory), the last state, the sum of the
        momenta values along the trajectory (needed for termination criterion),
        the updated termination state, the number of integration steps
        performed, a boolean that indicates whether the trajectory has diverged,
        a boolean that indicates whether the termination criterion was met. And
        on the other hand a dictionary that contains the update rules for the
        shared variables updated in the internal `Scan` operator.

        """

        def add_one_state(
            step,
            q_proposal,  # current proposal
            p_proposal,
            potential_energy_proposal,
            potential_energy_grad_proposal,
            energy_proposal,
            weight,
            sum_p_accept,
            q_last,  # state
            p_last,
            potential_energy_last,
            potential_energy_grad_last,
            momentum_sum: TensorVariable,  # sum of momenta
            momentum_ckpts,  # termination state
            momentum_sum_ckpts,
            idx_min,
            idx_max,
            trajectory_length,
        ):
            state = (
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
                sum_p_accept,
            )

            new_state = integrator(*state, direction * step_size)
            new_proposal, is_diverging = generate_proposal(initial_energy, new_state)
            sampled_proposal = sample_proposal(srng, proposal, new_proposal)

            new_momentum_sum = momentum_sum + new_state[1]
            new_termination_state = update_termination_state(
                termination_state, new_momentum_sum, new_state[1], step
            )
            has_terminated = is_criterion_met(
                new_termination_state, new_momentum_sum, new_state[1]
            )

            do_stop_integrating = is_diverging | has_terminated

            return (
                *sampled_proposal[0],
                sampled_proposal[1],
                sampled_proposal[2],
                sampled_proposal[3],
                *new_state,
                new_momentum_sum,
                *new_termination_state,
                trajectory_length + 1,
                is_diverging,
                has_terminated,
            ), until(do_stop_integrating)

        # We take one step away to start building the subtrajectory
        state = integrator(*previous_last_state, direction * step_size)
        proposal, is_diverging = generate_proposal(initial_energy, state)
        momentum_sum = state[1]
        termination_state = update_termination_state(
            termination_state,
            momentum_sum,
            state[1],
            0,
        )
        full_initial_state = (
            *proposal[0],
            proposal[1],
            proposal[2],
            proposal[3],
            *state,
            momentum_sum,
            *termination_state,
            at.as_tensor(1, dtype=np.int32),
            is_diverging,
            np.array(False),
        )

        steps = at.arange(1, 1 + max_num_steps)
        trajectory, updates = aesara.scan(
            add_one_state,
            outputs_info=(
                *proposal[0],
                proposal[1],
                proposal[2],
                proposal[3],
                *state,
                momentum_sum,
                *termination_state,
                at.as_tensor(1, dtype=np.int32),
                None,
                None,
            ),
            sequences=steps,
        )
        full_last_state = tuple([state[-1] for state in trajectory])

        # We build the trajectory iff the first step is not diverging
        full_state = ifelse(is_diverging, full_initial_state, full_last_state)

        new_proposal = (
            (full_state[0], full_state[1], full_state[2], full_state[3]),
            full_state[4],
            full_state[5],
            full_state[6],
        )
        new_state = (full_state[7], full_state[8], full_state[9], full_state[10])
        subtree_momentum_sum = full_state[11]
        new_termination_state = (
            full_state[12],
            full_state[13],
            full_state[14],
            full_state[15],
        )
        trajectory_length = full_state[-3]
        is_diverging = full_state[-2]
        has_terminated = full_state[-1]

        return (
            new_proposal,
            new_state,
            subtree_momentum_sum,
            new_termination_state,
            trajectory_length,
            is_diverging,
            has_terminated,
        ), updates

    return integrate


def multiplicative_expansion(
    srng: RandomStream,
    trajectory_integrator: Callable,
    uturn_check_fn: Callable,
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
    srng
        A RandomStream object that tracks the changes in a shared random state.
    trajectory_integrator
        A function that runs the symplectic integrators and returns a new proposal
        and the integrated trajectory.
    uturn_check_fn
        Function used to check the U-Turn criterion.
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
        step_size,
    ):
        """Expand the current trajectory multiplicatively.

        At each step we draw a direction at random, build a subtrajectory starting
        from the leftmost or rightmost point of the current trajectory that is
        twice as long as the current trajectory.

        Once that is done, possibly update the current proposal with that of
        the subtrajectory.

        Parameters
        ----------
        proposal
            Current new state proposal.
        left_state
            The current leftmost state of the trajectory.
        right_state
            The current rightmost state of the trajectory.
        momentum_sum
            The current value of the sum of momenta along the trajectory.
        initial_energy
            Potential energy before starting to build the trajectory.
        step_size
            The size of each step taken by the integrator.

        """

        def expand_once(
            step,
            q_proposal,  # proposal
            p_proposal,
            potential_energy_proposal,
            potential_energy_grad_proposal,
            energy_proposal,
            weight,
            sum_p_accept,
            q_left,  # trajectory
            p_left,
            potential_energy_left,
            potential_energy_grad_left,
            q_right,
            p_right,
            potential_energy_right,
            potential_energy_grad_right,
            momentum_sum,  # sum of momenta along trajectory
            momentum_ckpts,  # termination_state
            momentum_sum_ckpts,
            idx_min,
            idx_max,
        ):

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
                sum_p_accept,
            )
            termination_state = (
                momentum_ckpts,
                momentum_sum_ckpts,
                idx_min,
                idx_max,
            )

            do_go_right = srng.bernoulli(0.5)
            direction = at.where(do_go_right, 1.0, -1.0)
            start_state = ifelse(do_go_right, right_state, left_state)

            (
                new_proposal,
                new_state,
                subtree_momentum_sum,
                new_termination_state,
                subtrajectory_length,
                is_diverging,
                has_subtree_terminated,
            ), inner_updates = trajectory_integrator(
                start_state,
                direction,
                termination_state,
                2**step,
                step_size,
                initial_energy,
            )

            # Update the trajectory.
            # The trajectory integrator always integrates forward in time; we
            # thus need to switch the states if the other direction was picked.
            new_left_state = ifelse(do_go_right, left_state, new_state)
            new_right_state = ifelse(do_go_right, new_state, right_state)
            new_momentum_sum = momentum_sum + subtree_momentum_sum

            # Compute the pseudo-acceptance probability for the NUTS algorithm.
            # It can be understood as the average acceptance probability MC would give to
            # the states explored during the final expansion.
            acceptance_probability = at.exp(new_proposal[3]) / subtrajectory_length

            # Update the proposal
            #
            # We do not accept proposals that come from diverging or turning subtrajectories.
            # However the definition of the acceptance probability is such that the
            # acceptance probability needs to be computed across the entire trajectory.
            updated_proposal = (
                proposal[0],
                proposal[1],
                proposal[2],
                at.logaddexp(new_proposal[3], proposal[3]),
            )

            sampled_proposal = where_proposal(
                is_diverging | has_subtree_terminated,
                updated_proposal,
                proposal_sampler(srng, proposal, new_proposal),
            )

            # Check if the trajectory is turning and determine whether we need
            # to stop expanding the trajectory.
            is_turning = uturn_check_fn(
                new_left_state[1], new_right_state[1], new_momentum_sum
            )
            do_stop_expanding = is_diverging | is_turning | has_subtree_terminated

            return (
                (
                    *sampled_proposal[0],
                    sampled_proposal[1],
                    sampled_proposal[2],
                    sampled_proposal[3],
                    *new_left_state,
                    *new_right_state,
                    new_momentum_sum,
                    *new_termination_state,
                    acceptance_probability,
                    step + 1,
                    is_diverging,
                    is_turning,
                ),
                inner_updates,
                until(do_stop_expanding),
            )

        expansion_steps = at.arange(0, max_num_expansions)
        results, updates = aesara.scan(
            expand_once,
            outputs_info=(
                *proposal[0],
                proposal[1],
                proposal[2],
                proposal[3],
                *left_state,
                *right_state,
                momentum_sum,
                *termination_state,
                None,
                None,
                None,
                None,
            ),
            sequences=expansion_steps,
        )

        return results, updates

    return expand


def where_proposal(
    do_pick_left: bool,
    left_proposal: ProposalStateType,
    right_proposal: ProposalStateType,
) -> ProposalStateType:
    """Represents a switch between two proposals depending on a condition."""
    left_state, left_weight, left_energy, left_log_sum_p_accept = left_proposal
    right_state, right_weight, right_energy, right_log_sum_p_accept = right_proposal

    state = ifelse(do_pick_left, left_state, right_state)
    energy = at.where(do_pick_left, left_energy, right_energy)
    weight = at.where(do_pick_left, left_weight, right_weight)
    log_sum_p_accept = ifelse(
        do_pick_left, left_log_sum_p_accept, right_log_sum_p_accept
    )

    return (state, energy, weight, log_sum_p_accept)
