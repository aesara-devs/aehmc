from typing import Callable, NamedTuple, Tuple

import aesara.tensor as at
import numpy as np
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc.integrators import IntegratorState


class ProposalState(NamedTuple):
    state: IntegratorState
    energy: TensorVariable
    weight: TensorVariable
    sum_log_p_accept: TensorVariable


def proposal_generator(kinetic_energy: Callable, divergence_threshold: float):
    def update(initial_energy, state: IntegratorState) -> Tuple[ProposalState, bool]:
        """Generate a new proposal from a trajectory state.

        The trajectory state records information about the position in the state
        space and corresponding potential energy. A proposal also carries a
        weight that is equal to the difference between the current energy and
        the previous one. It thus carries information about the previous state
        as well as the current state.

        Parameters
        ----------
        initial_energy:
            The initial energy.
        state:
            The new state.

        Return
        ------
        A tuple that contains the new proposal and a boolean that indicates
        whether the current transition is divergent.

        """
        new_energy = state.potential_energy + kinetic_energy(state.momentum)

        delta_energy = initial_energy - new_energy
        delta_energy = at.where(at.isnan(delta_energy), -np.inf, delta_energy)
        is_transition_divergent = at.abs(delta_energy) > divergence_threshold

        weight = delta_energy
        log_p_accept = at.where(
            at.gt(delta_energy, 0),
            at.as_tensor(0, dtype=delta_energy.dtype),
            delta_energy,
        )

        return (
            ProposalState(
                state=state,
                energy=new_energy,
                weight=weight,
                sum_log_p_accept=log_p_accept,
            ),
            is_transition_divergent,
        )

    return update


# -------------------------------------------------------------------
#                     PROGRESSIVE SAMPLING
# -------------------------------------------------------------------


def progressive_uniform_sampling(
    srng: RandomStream, proposal: ProposalState, new_proposal: ProposalState
) -> ProposalState:
    """Uniform proposal sampling.

    Choose between the current proposal and the proposal built from the last
    trajectory state.

    Parameters
    ----------
    srng
        RandomStream object
    proposal
        The current proposal, it does not necessarily correspond to the
        previous state on the trajectory
    new_proposal
        The proposal built from the last trajectory state.

    Return
    ------
    Either the current or the new proposal.

    """
    # TODO: Make the `at.isnan` check unnecessary
    p_accept = at.expit(new_proposal.weight - proposal.weight)
    p_accept = at.where(at.isnan(p_accept), 0, p_accept)

    do_accept = srng.bernoulli(p_accept)
    updated_proposal = maybe_update_proposal(do_accept, proposal, new_proposal)

    return updated_proposal


def progressive_biased_sampling(
    srng: RandomStream, proposal: ProposalState, new_proposal: ProposalState
) -> ProposalState:
    """Baised proposal sampling.

    Choose between the current proposal and the proposal built from the last
    trajectory state. Unlike uniform sampling, biased sampling favors new
    proposals. It thus biases the transition away from the trajectory's initial
    state.

    Parameters
    ----------
    srng
        RandomStream object
    proposal
        The current proposal, it does not necessarily correspond to the
        previous state on the trajectory
    new_proposal
        The proposal built from the last trajectory state.

    Return
    ------
    Either the current or the new proposal.

    """
    p_accept = at.clip(at.exp(new_proposal.weight - proposal.weight), 0.0, 1.0)
    do_accept = srng.bernoulli(p_accept)
    updated_proposal = maybe_update_proposal(do_accept, proposal, new_proposal)

    return updated_proposal


def maybe_update_proposal(
    do_accept: bool, proposal: ProposalState, new_proposal: ProposalState
) -> ProposalState:
    """Return either proposal depending on the boolean `do_accept`"""
    updated_weight = at.logaddexp(proposal.weight, new_proposal.weight)
    updated_log_sum_p_accept = at.logaddexp(
        proposal.sum_log_p_accept, new_proposal.sum_log_p_accept
    )

    updated_q = at.where(
        do_accept, new_proposal.state.position, proposal.state.position
    )
    updated_p = at.where(
        do_accept, new_proposal.state.momentum, proposal.state.momentum
    )
    updated_potential_energy = at.where(
        do_accept, new_proposal.state.potential_energy, proposal.state.potential_energy
    )
    updated_potential_energy_grad = at.where(
        do_accept,
        new_proposal.state.potential_energy_grad,
        proposal.state.potential_energy_grad,
    )
    updated_energy = at.where(do_accept, new_proposal.energy, proposal.energy)

    updated_state = IntegratorState(
        position=updated_q,
        momentum=updated_p,
        potential_energy=updated_potential_energy,
        potential_energy_grad=updated_potential_energy_grad,
    )

    return ProposalState(
        state=updated_state,
        energy=updated_energy,
        weight=updated_weight,
        sum_log_p_accept=updated_log_sum_p_accept,
    )
