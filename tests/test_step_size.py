import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara import config
from aesara.tensor.random.utils import RandomStream

from aehmc import hmc
from aehmc.algorithms import DualAveragingState
from aehmc.step_size import dual_averaging_adaptation


@pytest.fixture()
def init():
    def logprob_fn(x):
        return -2 * (x - 1.0) ** 2

    srng = RandomStream(seed=0)
    kernel = hmc.new_kernel(srng, logprob_fn)

    initial_position = at.as_tensor(1.0, dtype=config.floatX)
    initial_state = hmc.new_state(initial_position, logprob_fn)

    return initial_state, kernel


def test_dual_averaging_adaptation(init):
    initial_state, kernel = init

    init_stepsize = at.as_tensor(1.0, dtype=config.floatX)
    inverse_mass_matrix = at.as_tensor(1.0)
    num_integration_steps = 10

    init_fn, update_fn = dual_averaging_adaptation()
    da_state = init_fn(init_stepsize)

    def one_step(q, logprob, logprob_grad, step, x_t, x_avg, gradient_avg, mu):
        state = hmc.IntegratorState(q, None, logprob, logprob_grad)
        (state, p_accept, _), inner_updates = kernel(
            state, at.exp(x_t), inverse_mass_matrix, num_integration_steps
        )
        current_da_state = DualAveragingState(
            step=step,
            iterates=x_t,
            iterates_avg=x_avg,
            gradient_avg=gradient_avg,
            shrinkage_pts=mu,
        )
        da_state = update_fn(p_accept, current_da_state)

        return (
            state.position,
            state.potential_energy,
            state.potential_energy_grad,
            da_state.step,
            da_state.iterates,
            da_state.iterates_avg,
            da_state.gradient_avg,
            da_state.shrinkage_pts,
            p_accept,
        ), inner_updates

    states, updates = aesara.scan(
        fn=one_step,
        outputs_info=[
            {"initial": initial_state.position},
            {"initial": initial_state.potential_energy},
            {"initial": initial_state.potential_energy_grad},
            {"initial": da_state.step},
            {"initial": da_state.iterates},  # logstepsize
            {"initial": da_state.iterates_avg},  # logstepsize_avg
            {"initial": da_state.gradient_avg},
            {"initial": da_state.shrinkage_pts},  # mu
            None,
        ],
        n_steps=10_000,
    )

    p_accept = aesara.function((), states[-1], updates=updates)
    step_size = aesara.function((), at.exp(states[-4][-1]), updates=updates)
    assert np.mean(p_accept()) == pytest.approx(0.8, rel=10e-3)
    assert step_size() < 10
    assert step_size() > 1e-1
