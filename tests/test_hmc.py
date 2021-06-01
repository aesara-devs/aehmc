from typing import Callable

import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

import aehmc.hmc as hmc


def normal_logp(q: TensorVariable):
    return aet.sum(aet.square(q - 3.0))


def build_trajectory_generator(
    srng: RandomStream,
    kernel_generator: Callable,
    potential_fn: Callable,
    num_states: int,
) -> Callable:
    q = aet.vector("q")
    potential_energy = potential_fn(q)
    potential_energy_grad = aesara.grad(potential_energy, wrt=q)

    step_size = aet.scalar("step_size")
    inverse_mass_matrix = aet.vector("inverse_mass_matrix")
    num_integration_steps = aet.scalar("num_integration_steps", dtype="int32")

    kernel = kernel_generator(
        srng, potential_fn, step_size, inverse_mass_matrix, num_integration_steps
    )

    trajectory, updates = aesara.scan(
        fn=kernel,
        outputs_info=[
            {"initial": q},
            {"initial": potential_energy},
            {"initial": potential_energy_grad},
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

    trajectory_generator = build_trajectory_generator(
        srng, hmc.kernel, normal_logp, 10_000
    )
    positions, *_ = trajectory_generator(
        initial_position, step_size, inverse_mass_matrix, num_integration_steps
    )

    assert np.mean(positions[9000:], axis=0) == pytest.approx(3, 1e-1)
