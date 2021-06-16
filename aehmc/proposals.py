from typing import Callable, Tuple

import aesara.tensor as aet
import numpy as np
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc.integrators import IntegratorStateType

ProposalStateType = Tuple[
    IntegratorStateType, TensorVariable, TensorVariable, TensorVariable
]


def proposal_generator(kinetic_energy: Callable, divergence_threshold: float):
    def init(state):
        _, p, potential_energy, _ = state
        energy = potential_energy + kinetic_energy(p)
        return (state, energy, 0, -np.inf)

    def update(initial_energy, state):
        q, p, potential_energy, _ = state
        new_energy = potential_energy + kinetic_energy(p)

        delta_energy = initial_energy - new_energy
        delta_energy = aet.where(aet.isnan(delta_energy), -np.inf, delta_energy)
        is_transition_divergent = aet.abs_(delta_energy) > divergence_threshold

        weight = delta_energy
        sum_log_p_accept = aet.minimum(delta_energy, 0.0)

        return (state, new_energy, weight, sum_log_p_accept), is_transition_divergent

    return init, update


# -------------------------------------------------------------------
#                     PROGRESSIVE SAMPLING
# -------------------------------------------------------------------


def progressive_uniform_sampling(
    srng: RandomStream, proposal: ProposalStateType, new_proposal: ProposalStateType
) -> ProposalStateType:
    state, energy, weight, sum_log_p_accept = proposal
    new_state, new_energy, new_weight, new_sum_log_p_accept = new_proposal

    p_accept = aet.expit(new_weight - weight)
    do_accept = srng.bernoulli(p_accept)
    updated_proposal = maybe_update_proposal(do_accept, proposal, new_proposal)

    return updated_proposal


def maybe_update_proposal(
    do_accept: bool, proposal: ProposalStateType, new_proposal: ProposalStateType
) -> ProposalStateType:
    state, energy, weight, sum_log_p_accept = proposal
    new_state, new_energy, new_weight, new_sum_log_p_accept = new_proposal

    updated_weight = _logaddexp(weight, new_weight)
    updated_sum_log_p_accept = _logaddexp(sum_log_p_accept, new_sum_log_p_accept)

    updated_q = aet.where(do_accept, new_state[0], state[0])
    updated_p = aet.where(do_accept, new_state[1], state[1])
    updated_potential_energy = aet.where(do_accept, new_state[2], state[2])
    updated_potential_energy_grad = aet.where(do_accept, new_state[3], state[3])
    updated_energy = aet.where(do_accept, new_energy, energy)

    return (
        (updated_q, updated_p, updated_potential_energy, updated_potential_energy_grad),
        updated_energy,
        updated_weight,
        updated_sum_log_p_accept,
    )


def _logaddexp(a, b):
    diff = b - a
    return aet.switch(
        diff > 0, b + aet.log1p(aet.exp(-diff)), a + aet.log1p(aet.exp(-diff))
    )
