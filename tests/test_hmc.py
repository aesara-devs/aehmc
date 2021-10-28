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
    step_size = 0.04
    inverse_mass_matrix = at.as_tensor(1.0)
    num_integration_steps = 20

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
        n_steps=20_000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory[0], updates=updates)

    samples = trajectory_generator(3.0)
    assert np.mean(samples[1000:]) == pytest.approx(1.0, rel=1e-1)
    assert np.var(samples[1000:]) == pytest.approx(4.0, rel=1e-1)


def logprob(q):
    return -at.sum(0.5 * at.square((q - 3) / 2))


def test_nuts():
    """Test the NUTS kernel on a gaussian target."""
    srng = RandomStream(seed=59)

    step_size = at.scalar("step_size", dtype="float64")
    inverse_mass_matrix = at.vector("inverse_mass_matrix", dtype="float64")
    kernel = nuts.kernel(srng, normal_logprob, inverse_mass_matrix)

    q = at.vector("q")
    initial_state = nuts.new_state(q, normal_logprob)

    trajectory, updates = aesara.scan(
        fn=kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            None,
            None,
        ],
        non_sequences=step_size,
        n_steps=1000,
    )

    trajectory_generator = aesara.function(
        (q, step_size, inverse_mass_matrix),
        trajectory,
        updates=updates,
    )

    step_size = 0.01
    inverse_mass_matrix = np.array([1.0])
    initial_position = np.array([1.0])
    trajectory = trajectory_generator(initial_position, step_size, inverse_mass_matrix)
    samples = trajectory[0][300:]

    assert np.mean(np.array(samples)) == pytest.approx(3, 1e-1)
    print(np.var(samples))
