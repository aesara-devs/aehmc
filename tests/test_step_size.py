import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream

from aehmc import hmc
from aehmc.step_size import dual_averaging_adaptation, heuristic_adaptation


def test_heuristic_adaptation():
    def logprob_fn(x):
        return -at.sum(0.5 * x)

    srng = RandomStream(seed=0)

    initial_position = at.as_tensor(1.0, dtype="floatX")
    logprob = logprob_fn(initial_position)
    logprob_grad = aesara.grad(logprob, wrt=initial_position)
    reference_state = (initial_position, logprob, logprob_grad)

    inverse_mass_matrix = at.as_tensor(1.0)

    kernel = hmc.kernel(srng, logprob_fn, inverse_mass_matrix, 10)

    epsilon_1 = heuristic_adaptation(
        kernel, reference_state, at.as_tensor(1, dtype="floatX"), 0.95
    )
    epsilon_1_val = epsilon_1.eval()
    assert epsilon_1_val != np.inf

    epsilon_2 = heuristic_adaptation(
        kernel, reference_state, at.as_tensor(1, dtype="floatX"), 0.05
    )
    epsilon_2_val = epsilon_2.eval()
    assert epsilon_2_val > epsilon_1_val
    assert epsilon_2_val != np.inf


def test_dual_averaging_adaptation():
    def logprob_fn(x):
        return -at.sum(0.5 * x ** 2)

    srng = RandomStream(seed=0)
    inverse_mass_matrix = at.as_tensor(1.0)
    kernel = hmc.kernel(srng, logprob_fn, inverse_mass_matrix, 10)

    initial_position = at.as_tensor(1.0, dtype="floatX")
    logprob = logprob_fn(initial_position)
    logprob_grad = aesara.grad(logprob, wrt=initial_position)

    step_size = at.as_tensor(1.0, dtype="floatX")
    logpstepsize = at.log(step_size)
    init, update = dual_averaging_adaptation(step_size)
    step, logstepsize_avg, gradient_avg = init(at.as_tensor(0.0, dtype="floatX"))

    def one_step(q, logprob, logprob_grad, step, x_t, x_avg, gradient_avg):
        *state, p_accept = kernel(q, logprob, logprob_grad, at.exp(x_t))
        da_state = update(p_accept, step, x_t, x_avg, gradient_avg)
        return (*state, *da_state, p_accept)

    states, updates = aesara.scan(
        fn=one_step,
        outputs_info=[
            {"initial": initial_position},
            {"initial": logprob},
            {"initial": logprob_grad},
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
    assert step_size() > 10e-1
