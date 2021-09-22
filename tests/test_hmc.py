from typing import Callable

import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc import hmc, nuts


def normal_logprob(q: TensorVariable):
    return -aet.sum(aet.square(q - 3.0))


def normal_potential(q: TensorVariable):
    return -normal_logprob(q)


def build_hmc_trajectory_generator(
    srng: RandomStream,
    kernel_generator: Callable,
    state_generator: Callable,
    logprob_fn: Callable,
    num_states: int,
) -> Callable:
    q = aet.vector("q")
    initial_state = state_generator(q, logprob_fn)

    step_size = aet.scalar("step_size")
    inverse_mass_matrix = aet.vector("inverse_mass_matrix")
    num_integration_steps = aet.scalar("num_integration_steps", dtype="int32")

    kernel = kernel_generator(
        srng, logprob_fn, step_size, inverse_mass_matrix, num_integration_steps
    )

    trajectory, updates = aesara.scan(
        fn=kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
        ],
        n_steps=num_states,
    )
    trajectory_generator = aesara.function(
        (q, step_size, inverse_mass_matrix, num_integration_steps),
        trajectory,
        updates=updates,
    )

    return trajectory_generator


def test_hmc():
    """Test the HMC kernel on a simple potential."""
    srng = RandomStream(seed=59)
    step_size = 0.003
    num_integration_steps = 10
    initial_position = np.array([1.0])
    inverse_mass_matrix = np.array([1.0])

    trajectory_generator = build_hmc_trajectory_generator(
        srng, hmc.kernel, hmc.new_state, normal_logprob, 50_000
    )
    positions, *_ = trajectory_generator(
        initial_position, step_size, inverse_mass_matrix, num_integration_steps
    )

    assert np.mean(positions[10_000:], axis=0) == pytest.approx(3, 1e-1)


def test_nuts():
    """Test the NUTS kernel on a simple potential."""
    srng = RandomStream(seed=59)

    q = aet.vector("q")
    initial_state = nuts.new_state(q, normal_logprob)

    step_size = aet.scalar("step_size", dtype="float64")
    inverse_mass_matrix = aet.vector("inverse_mass_matrix", dtype="float64")

    kernel = nuts.kernel(srng, normal_logprob, step_size, inverse_mass_matrix)
    result, updates = kernel(*initial_state)

    trajectory_generator = aesara.function(
        (q, step_size, inverse_mass_matrix),
        result,
        updates=updates,
        allow_input_downcast=True,
    )

    step_size = 0.01
    initial_position = np.array([1.0])
    inverse_mass_matrix = np.array([1.0])

    q = initial_position
    positions = []
    for _ in range(1000):
        result = trajectory_generator(q, step_size, inverse_mass_matrix)
        q = result[0]
        positions.append(q)

    assert np.mean(np.array(positions)[300:]) == pytest.approx(3, 1e-1)
