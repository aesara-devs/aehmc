import aesara
import aesara.tensor as aet
import numpy as np
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

import aehmc.hmc as hmc


def test_hmc():
    def potential_fn(q: TensorVariable) -> TensorVariable:
        return -1.0 / aet.power(aet.square(q[0]) + aet.square(q[1]), 0.5)

    srng = RandomStream(seed=59)

    step_size = aet.scalar("step_size")
    inverse_mass_matrix = aet.vector("inverse_mass_matrix")
    num_integration_steps = aet.scalar("num_integration_steps", dtype="int32")

    q = aet.vector("q")
    potential_energy = potential_fn(q)
    potential_energy_grad = aesara.grad(potential_energy, wrt=q)

    kernel = hmc.kernel(
        srng, potential_fn, step_size, inverse_mass_matrix, num_integration_steps
    )
    next_step = kernel(q, potential_energy, potential_energy_grad)

    # Compile a function that returns the next state
    step_fn = aesara.function(
        (q, step_size, inverse_mass_matrix, num_integration_steps), next_step
    )

    # Compile a function that integrates the trajectory integrating several times
    trajectory, _ = aesara.scan(
        fn=kernel,
        outputs_info=[
            {"initial": q},
            {"initial": potential_energy},
            {"initial": potential_energy_grad},
        ],
        n_steps=1000,
    )
    traj = aesara.function(
        (q, step_size, inverse_mass_matrix, num_integration_steps), trajectory
    )

    # Run
    step_size = 0.01
    num_integration_steps = 3
    q = np.array([1.0, 1.0])
    inverse_mass_matrix = np.array([1.0, 1.0])

    # This works
    samples = []
    for _ in range(1_000):
        q, *_ = step_fn(q, step_size, inverse_mass_matrix, num_integration_steps)
        samples.append(q)
    print(np.mean(np.array(samples)))

    # This doesn't
    print(traj(q, step_size, inverse_mass_matrix, num_integration_steps))
