import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aeppl import joint_logprob
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc import hmc, nuts


def normal_logprob(q: TensorVariable):
    y = (q - 3.0) / 5.0
    return -at.sum(at.square(y))


def test_hmc():
    """Test the HMC kernel on a gaussian target."""
    step_size = 1.0
    inverse_mass_matrix = at.as_tensor(1.0)
    num_integration_steps = 10

    Y_rv = at.random.normal(1, 2)

    def logprob_fn(y):
        logprob = joint_logprob({Y_rv: y})
        return logprob

    srng = RandomStream(seed=0)
    kernel = hmc.kernel(
        srng,
        logprob_fn,
        inverse_mass_matrix,
        num_integration_steps,
    )

    y_vv = Y_rv.clone()
    initial_state = hmc.new_state(y_vv, logprob_fn)

    trajectory, updates = aesara.scan(
        kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            None,
        ],
        non_sequences=step_size,
        n_steps=2_000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory[0], updates=updates)

    samples = trajectory_generator(3.0)
    assert np.mean(samples[1000:]) == pytest.approx(1.0, rel=1e-1)
    assert np.var(samples[1000:]) == pytest.approx(4.0, rel=1e-1)


def test_nuts():
    """Test the NUTS kernel on a gaussian target."""
    step_size = 1.0
    inverse_mass_matrix = at.as_tensor(1.0)

    Y_rv = at.random.normal(1, 2)

    def logprob_fn(y):
        logprob = joint_logprob({Y_rv: y})
        return logprob

    srng = RandomStream(seed=0)
    kernel = nuts.kernel(
        srng,
        logprob_fn,
        inverse_mass_matrix,
    )

    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    trajectory, updates = aesara.scan(
        kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            None,
            None,
            None,
            None,
        ],
        non_sequences=step_size,
        n_steps=2000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory[0], updates=updates)

    samples = trajectory_generator(3.0)
    assert np.mean(samples[1000:]) == pytest.approx(1.0, rel=1e-1)
    assert np.var(samples[1000:]) == pytest.approx(4.0, rel=1e-1)
