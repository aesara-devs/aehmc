import aesara
import aesara.tensor as at
import numpy as np
from aesara.tensor.random.utils import RandomStream

from aehmc import hmc
from aehmc.step_size import heuristic_adaptation


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
