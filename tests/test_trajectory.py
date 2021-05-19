import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.var import TensorVariable

from aesara_hmc.integrators import velocity_verlet
from aesara_hmc.trajectory import static_integration


def CircularMotion(inverse_mass_matrix):
    def potential_energy(q: TensorVariable) -> TensorVariable:
        return -1.0 / aet.power(aet.square(q[0]) + aet.square(q[1]), 0.5)

    def kinetic_energy(p: TensorVariable) -> TensorVariable:
        return 0.5 * aet.dot(inverse_mass_matrix, aet.square(p))

    return potential_energy, kinetic_energy


examples = [
    {
        "n_steps": 628,
        "step_size": 0.01,
        "q_init": np.array([1.0, 0.0]),
        "p_init": np.array([0.0, 1.0]),
        "q_final": np.array([1.0, 0.0]),
        "p_final": np.array([0.0, 1.0]),
        "inverse_mass_matrix": np.array([1.0, 1.0]),
    },
]


@pytest.mark.parametrize("example", examples)
def test_static_integration(example):
    inverse_mass_matrix = example["inverse_mass_matrix"]
    step_size = example["step_size"]
    num_steps = example["n_steps"]
    q_init = example["q_init"]
    p_init = example["p_init"]

    potential, kinetic_energy = CircularMotion(inverse_mass_matrix)
    step = velocity_verlet(potential, kinetic_energy)
    integrator = static_integration(step, step_size, num_steps)

    q = aet.vector("q")
    p = aet.vector("p")
    energy = potential(q)
    energy_grad = aesara.grad(energy, q)
    final_state = integrator(q, p, energy, energy_grad)
    integrate_fn = aesara.function((q, p), final_state)

    q_final, p_final, *_ = integrate_fn(q_init, p_init)

    np.testing.assert_allclose(q_final, example["q_final"], atol=1e-1)
    np.testing.assert_allclose(p_final, example["p_final"], atol=1e-1)
