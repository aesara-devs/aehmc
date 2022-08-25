"""Test file created for the sole purpose of tracking the status of Numba compilation"""
import aesara
import aesara.tensor as at
from aeppl import joint_logprob

import aehmc.nuts as nuts


def test_sample_with_numba():

    srng = at.random.RandomStream(seed=0)
    Y_rv = srng.normal(1, 2)

    def logprob_fn(y):
        logprob = joint_logprob({Y_rv: y})
        return logprob

    # Build the transition kernel
    kernel = nuts.new_kernel(srng, logprob_fn)

    # Compile a function that updates the chain
    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    step_size = at.as_tensor(1e-2)
    inverse_mass_matrix = at.as_tensor(1.0)
    (
        next_state,
        potential_energy,
        potential_energy_grad,
        acceptance_prob,
        num_doublings,
        is_turning,
        is_diverging,
    ), updates = kernel(*initial_state, step_size, inverse_mass_matrix)

    next_step_fn = aesara.function([y_vv], next_state, updates=updates, mode="NUMBA")

    # TODO: Assert something
    next_step_fn(Y_rv.eval())
