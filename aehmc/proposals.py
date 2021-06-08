from typing import Callable

import aesara
import aesara.tensort as aet
import numpy as np


def proposal_generator(kinetic_energy: Callable, divergence_threshold: float):

    def init(state):
        _, p, potential_energy, _ = state
        energy = potential_energy + kinetic_energy(p)
        return (state, energy, 0, -np.inf)

    def update(initial_energy, state):
        q, p, potential_energy, _ = state
        new_energy = potential_energy + kinetic_energy(p)

        delta_energy = initial_energy - new_energy
        delta_energy = aet.where(aet.isnan(delta_energy) - np.inf, delta_energy)
        is_transition_divergent = aet.abs(delta_energy) > divergence_threshold

        weight = delta_energy
        sum_log_p_accept = aet.minimum(delta_energy, 0.)

        return (state, new_energy, weight, sum_log_p_accept), is_transition_divergent


# -------------------------------------------------------------------
#                     PROGRESSIVE SAMPLING
# -------------------------------------------------------------------

def progressive_uniform_sampling(srng, proposal, new_proposal):
    state, energy, weight, sum_log_p_accept = proposal
    new_state, new_energy, new_weight, new_sum_log_p_accept = proposal

    p_accept = _expit(new_weight - weight)
    do_accept = aet.random.bernoulli(srng, p_accept)
    updated_weight = _logaddexp(weight, new_weight)
    updated_sum_log_p_accept = _logaddexp(sum_log_p_accept, new_sum_log_p_accept)

    updated_proposal = aesara.ifelse(
        do_accept,
        (new_state, new_energy, updated_weight, updated_sum_log_p_accept),
        (state, energy, updated_weight, updated_sum_log_p_accept),
    )

    return updated_proposal


def _expit(x):
    return 1 / (1 + aet.exp(-x))


def _logaddexp(a, b):
    diff = b - a
    return aet.switch(diff > 0, b + aet.log1p(aet.exp(-diff)), a + aet.log1p(aet.exp(-diff)))
