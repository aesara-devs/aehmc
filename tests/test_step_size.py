import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream

from aehmc import hmc
from aehmc.step_size import dual_averaging_adaptation, heuristic_adaptation


@pytest.fixture()
def init():
    def logprob_fn(x):
        return -2 * (x - 1.0) ** 2

    srng = RandomStream(seed=0)
    inverse_mass_matrix = at.as_tensor(1.0)
    kernel = hmc.kernel(srng, logprob_fn, inverse_mass_matrix, 10)

    initial_position = at.as_tensor(1.0, dtype="floatX")
    initial_state = hmc.new_state(initial_position, logprob_fn)

    return initial_state, kernel


def test_heuristic_adaptation(init):
    reference_state, kernel = init

    epsilon_1 = heuristic_adaptation(
        kernel, reference_state, at.as_tensor(0.5, dtype="floatX"), 0.95
    )
    epsilon_1_val = epsilon_1.eval()
    assert epsilon_1_val != np.inf

    epsilon_2 = heuristic_adaptation(
        kernel, reference_state, at.as_tensor(0.5, dtype="floatX"), 0.05
    )
    epsilon_2_val = epsilon_2.eval()
    assert epsilon_2_val > epsilon_1_val
    assert epsilon_2_val != np.inf


def test_dual_averaging_adaptation(init):
    initial_state, kernel = init

    logpstepsize = at.log(at.as_tensor(1.0, dtype="floatX"))
    init, update = dual_averaging_adaptation(logpstepsize)
    step, logstepsize_avg, gradient_avg = init(at.as_tensor(0.0, dtype="floatX"))

    def one_step(q, logprob, logprob_grad, step, x_t, x_avg, gradient_avg):
        *state, p_accept = kernel(q, logprob, logprob_grad, at.exp(x_t))
        da_state = update(p_accept, step, x_t, x_avg, gradient_avg)
        return (*state, *da_state, p_accept)

    states, updates = aesara.scan(
        fn=one_step,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            {"initial": step},
            {"initial": logpstepsize},
            {"initial": logstepsize_avg},
            {"initial": gradient_avg},
            None,
        ],
        n_steps=10_000,
    )

    p_accept = aesara.function((), states[-1], updates=updates)
    step_size = aesara.function((), at.exp(states[-3][-1]), updates=updates)
    assert np.mean(p_accept()) == pytest.approx(0.65, rel=10e-3)
    assert step_size() < 10
    assert step_size() > 1e-1
