from typing import Callable, Dict, NamedTuple, Tuple

import aesara
import aesara.tensor as at
import numpy as np
from aesara.ifelse import ifelse
from aesara.scan.utils import until
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc.integrators import IntegratorState
from aehmc.proposals import (
    ProposalState,
    progressive_biased_sampling,
    progressive_uniform_sampling,
    proposal_generator,
)
from aehmc.termination import TerminationState

__all__ = ["static_integration", "dynamic_integration", "multiplicative_expansion"]


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
        init_state: IntegratorState, step_size: TensorVariable
    ) -> Tuple[IntegratorState, Dict]:
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
                IntegratorState(q, p, potential_energy, potential_energy_grad),
                step_size,
            )
            return new_state

        [q, p, energy, energy_grad], updates = aesara.scan(
            fn=one_step,
            outputs_info=[
                {"initial": init_state.position},
                {"initial": init_state.momentum},
                {"initial": init_state.potential_energy},
                {"initial": init_state.potential_energy_grad},
            ],
            n_steps=num_integration_steps,
        )

        return (
            IntegratorState(
                position=q[-1],
                momentum=p[-1],
                potential_energy=energy[-1],
                potential_energy_grad=energy_grad[-1],
            ),
            updates,
        )

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
        previous_last_state: IntegratorState,
        direction: TensorVariable,
        termination_state: TerminationState,
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
            termination_state = TerminationState(
                momentum_checkpoints=momentum_ckpts,
                momentum_sum_checkpoints=momentum_sum_ckpts,
                min_index=idx_min,
                max_index=idx_max,
            )
            proposal = ProposalState(
                state=IntegratorState(
                    position=q_proposal,
                    momentum=p_proposal,
                    potential_energy=potential_energy_proposal,
                    potential_energy_grad=potential_energy_grad_proposal,
                ),
                energy=energy_proposal,
                weight=weight,
                sum_log_p_accept=sum_p_accept,
            )
            last_state = IntegratorState(
                position=q_last,
                momentum=p_last,
                potential_energy=potential_energy_last,
                potential_energy_grad=potential_energy_grad_last,
            )

            new_state = integrator(last_state, direction * step_size)
            new_proposal, is_diverging = generate_proposal(initial_energy, new_state)
            sampled_proposal = sample_proposal(srng, proposal, new_proposal)

            new_momentum_sum = momentum_sum + new_state.momentum
            new_termination_state = update_termination_state(
                termination_state, new_momentum_sum, new_state.momentum, step
            )
            has_terminated = is_criterion_met(
                new_termination_state, new_momentum_sum, new_state.momentum
            )

            do_stop_integrating = is_diverging | has_terminated

            return (
                sampled_proposal.state.position,
                sampled_proposal.state.momentum,
                sampled_proposal.state.potential_energy,
                sampled_proposal.state.potential_energy_grad,
                sampled_proposal.energy,
                sampled_proposal.weight,
                sampled_proposal.sum_log_p_accept,
                new_state.position,
                new_state.momentum,
                new_state.potential_energy,
                new_state.potential_energy_grad,
                new_momentum_sum,
                new_termination_state.momentum_checkpoints,
                new_termination_state.momentum_sum_checkpoints,
                new_termination_state.min_index,
                new_termination_state.max_index,
                trajectory_length + 1,
                is_diverging,
                has_terminated,
            ), until(do_stop_integrating)

        # We take one step away to start building the subtrajectory
        state = integrator(previous_last_state, direction * step_size)
        proposal, is_diverging = generate_proposal(initial_energy, state)
        momentum_sum = state.momentum
        termination_state = update_termination_state(
            termination_state,
            momentum_sum,
            state.momentum,
            0,
        )
        full_initial_state = (
            proposal.state.position,
            proposal.state.momentum,
            proposal.state.potential_energy,
            proposal.state.potential_energy_grad,
            proposal.energy,
            proposal.weight,
            proposal.sum_log_p_accept,
            state.position,
            state.momentum,
            state.potential_energy,
            state.potential_energy_grad,
            momentum_sum,
            termination_state.momentum_checkpoints,
            termination_state.momentum_sum_checkpoints,
            termination_state.min_index,
            termination_state.max_index,
            at.as_tensor(1, dtype=np.int64),
            is_diverging,
            np.array(False),
        )

        steps = at.arange(1, 1 + max_num_steps)
        trajectory, updates = aesara.scan(
            add_one_state,
            outputs_info=(
                proposal.state.position,
                proposal.state.momentum,
                proposal.state.potential_energy,
                proposal.state.potential_energy_grad,
                proposal.energy,
                proposal.weight,
                proposal.sum_log_p_accept,
                state.position,
                state.momentum,
                state.potential_energy,
                state.potential_energy_grad,
                momentum_sum,
                termination_state.momentum_checkpoints,
                termination_state.momentum_sum_checkpoints,
                termination_state.min_index,
                termination_state.max_index,
                at.as_tensor(1, dtype=np.int64),
                None,
                None,
            ),
            sequences=steps,
        )
        full_last_state = tuple([_state[-1] for _state in trajectory])

        # We build the trajectory iff the first step is not diverging
        full_state = ifelse(is_diverging, full_initial_state, full_last_state)

        new_proposal = ProposalState(
            state=IntegratorState(
                position=full_state[0],
                momentum=full_state[1],
                potential_energy=full_state[2],
                potential_energy_grad=full_state[3],
            ),
            energy=full_state[4],
            weight=full_state[5],
            sum_log_p_accept=full_state[6],
        )
        new_state = IntegratorState(
            position=full_state[7],
            momentum=full_state[8],
            potential_energy=full_state[9],
            potential_energy_grad=full_state[10],
        )
        subtree_momentum_sum = full_state[11]
        new_termination_state = TerminationState(
            momentum_checkpoints=full_state[12],
            momentum_sum_checkpoints=full_state[13],
            min_index=full_state[14],
            max_index=full_state[15],
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


class Diagnostics(NamedTuple):
    state: IntegratorState
    acceptance_probability: TensorVariable
    num_doublings: TensorVariable
    is_turning: TensorVariable
    is_diverging: TensorVariable


class MultiplicativeExpansionResult(NamedTuple):
    proposals: ProposalState
    right_states: IntegratorState
    left_states: IntegratorState
    momentum_sums: TensorVariable
    termination_states: TerminationState
    diagnostics: Diagnostics


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
        proposal: ProposalState,
        left_state: IntegratorState,
        right_state: IntegratorState,
        momentum_sum,
        termination_state: TerminationState,
        initial_energy,
        step_size,
    ) -> Tuple[MultiplicativeExpansionResult, Dict]:
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
        ) -> Tuple[Tuple[TensorVariable, ...], Dict, until]:
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
            proposal = ProposalState(
                state=IntegratorState(
                    position=q_proposal,
                    momentum=p_proposal,
                    potential_energy=potential_energy_proposal,
                    potential_energy_grad=potential_energy_grad_proposal,
                ),
                energy=energy_proposal,
                weight=weight,
                sum_log_p_accept=sum_p_accept,
            )
            termination_state = TerminationState(
                momentum_checkpoints=momentum_ckpts,
                momentum_sum_checkpoints=momentum_sum_ckpts,
                min_index=idx_min,
                max_index=idx_max,
            )

            do_go_right = srng.bernoulli(0.5)
            direction = at.where(do_go_right, 1.0, -1.0)
            start_state = IntegratorState(*ifelse(do_go_right, right_state, left_state))

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
            new_left_state = IntegratorState(
                *ifelse(do_go_right, left_state, new_state)
            )
            new_right_state = IntegratorState(
                *ifelse(do_go_right, new_state, right_state)
            )
            new_momentum_sum = momentum_sum + subtree_momentum_sum

            # Compute the pseudo-acceptance probability for the NUTS algorithm.
            # It can be understood as the average acceptance probability MC would give to
            # the states explored during the final expansion.
            acceptance_probability = (
                at.exp(new_proposal.sum_log_p_accept) / subtrajectory_length
            )

            # Update the proposal
            #
            # We do not accept proposals that come from diverging or turning subtrajectories.
            # However the definition of the acceptance probability is such that the
            # acceptance probability needs to be computed across the entire trajectory.
            updated_proposal = proposal._replace(
                sum_log_p_accept=at.logaddexp(
                    new_proposal.sum_log_p_accept, proposal.sum_log_p_accept
                )
            )

            sampled_proposal = where_proposal(
                is_diverging | has_subtree_terminated,
                updated_proposal,
                proposal_sampler(srng, proposal, new_proposal),
            )

            # Check if the trajectory is turning and determine whether we need
            # to stop expanding the trajectory.
            is_turning = uturn_check_fn(
                new_left_state.momentum, new_right_state.momentum, new_momentum_sum
            )
            do_stop_expanding = is_diverging | is_turning | has_subtree_terminated

            return (
                (
                    sampled_proposal.state.position,
                    sampled_proposal.state.momentum,
                    sampled_proposal.state.potential_energy,
                    sampled_proposal.state.potential_energy_grad,
                    sampled_proposal.energy,
                    sampled_proposal.weight,
                    sampled_proposal.sum_log_p_accept,
                    new_left_state.position,
                    new_left_state.momentum,
                    new_left_state.potential_energy,
                    new_left_state.potential_energy_grad,
                    new_right_state.position,
                    new_right_state.momentum,
                    new_right_state.potential_energy,
                    new_right_state.potential_energy_grad,
                    new_momentum_sum,
                    new_termination_state.momentum_checkpoints,
                    new_termination_state.momentum_sum_checkpoints,
                    new_termination_state.min_index,
                    new_termination_state.max_index,
                    acceptance_probability,
                    step + 1,
                    is_diverging,
                    is_turning,
                ),
                inner_updates,
                until(do_stop_expanding),
            )

        expansion_steps = at.arange(0, max_num_expansions)
        # results, updates = aesara.scan(
        (
            proposal_state_position,
            proposal_state_momentum,
            proposal_state_potential_energy,
            proposal_state_potential_energy_grad,
            proposal_energy,
            proposal_weight,
            proposal_sum_log_p_accept,
            left_state_position,
            left_state_momentum,
            left_state_potential_energy,
            left_state_potential_energy_grad,
            right_state_position,
            right_state_momentum,
            right_state_potential_energy,
            right_state_potential_energy_grad,
            momentum_sum,
            momentum_checkpoints,
            momentum_sum_checkpoints,
            min_indices,
            max_indices,
            acceptance_probability,
            num_doublings,
            is_diverging,
            is_turning,
        ), updates = aesara.scan(
            expand_once,
            outputs_info=(
                proposal.state.position,
                proposal.state.momentum,
                proposal.state.potential_energy,
                proposal.state.potential_energy_grad,
                proposal.energy,
                proposal.weight,
                proposal.sum_log_p_accept,
                left_state.position,
                left_state.momentum,
                left_state.potential_energy,
                left_state.potential_energy_grad,
                right_state.position,
                right_state.momentum,
                right_state.potential_energy,
                right_state.potential_energy_grad,
                momentum_sum,
                termination_state.momentum_checkpoints,
                termination_state.momentum_sum_checkpoints,
                termination_state.min_index,
                termination_state.max_index,
                None,
                None,
                None,
                None,
            ),
            sequences=expansion_steps,
        )
        # Ensure each item of the returned result sequence is packed into the appropriate namedtuples.
        typed_result = MultiplicativeExpansionResult(
            proposals=ProposalState(
                state=IntegratorState(
                    position=proposal_state_position,
                    momentum=proposal_state_momentum,
                    potential_energy=proposal_state_potential_energy,
                    potential_energy_grad=proposal_state_potential_energy_grad,
                ),
                energy=proposal_energy,
                weight=proposal_weight,
                sum_log_p_accept=proposal_sum_log_p_accept,
            ),
            left_states=IntegratorState(
                position=left_state_position,
                momentum=left_state_momentum,
                potential_energy=left_state_potential_energy,
                potential_energy_grad=left_state_potential_energy_grad,
            ),
            right_states=IntegratorState(
                position=right_state_position,
                momentum=right_state_momentum,
                potential_energy=right_state_potential_energy,
                potential_energy_grad=right_state_potential_energy_grad,
            ),
            momentum_sums=momentum_sum,
            termination_states=TerminationState(
                momentum_checkpoints=momentum_checkpoints,
                momentum_sum_checkpoints=momentum_sum_checkpoints,
                min_index=min_indices,
                max_index=max_indices,
            ),
            diagnostics=Diagnostics(
                state=IntegratorState(
                    position=proposal_state_position,
                    momentum=proposal_state_momentum,
                    potential_energy=proposal_state_potential_energy,
                    potential_energy_grad=proposal_state_potential_energy_grad,
                ),
                acceptance_probability=acceptance_probability,
                num_doublings=num_doublings,
                is_turning=is_turning,
                is_diverging=is_diverging,
            ),
        )
        return typed_result, updates

    return expand


def where_proposal(
    do_pick_left: bool,
    left_proposal: ProposalState,
    right_proposal: ProposalState,
) -> ProposalState:
    """Represents a switch between two proposals depending on a condition."""
    state = ifelse(do_pick_left, left_proposal.state, right_proposal.state)
    energy = at.where(do_pick_left, left_proposal.energy, right_proposal.energy)
    weight = at.where(do_pick_left, left_proposal.weight, right_proposal.weight)
    log_sum_p_accept = ifelse(
        do_pick_left, left_proposal.sum_log_p_accept, right_proposal.sum_log_p_accept
    )

    return ProposalState(
        state=IntegratorState(*state),
        energy=energy,
        weight=weight,
        sum_log_p_accept=log_sum_p_accept,
    )
