import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.var import TensorVariable

from aesara_hmc.integrators import velocity_verlet


def HarmonicOscillator(inverse_mass_matrix, k=1.0, m=1.0):
    """Potential and Kinetic energy of an harmonic oscillator."""

    def potential_energy(x: TensorVariable) -> TensorVariable:
        return aet.sum(0.5 * k * aet.square(x))

    def kinetic_energy(p: TensorVariable) -> TensorVariable:
        v = inverse_mass_matrix * p
        return aet.sum(0.5 * aet.dot(v, p))

    return potential_energy, kinetic_energy


def FreeFall(inverse_mass_matrix, g=1.0):
    """Potential and kinetic energy of a free-falling object."""

    def potential_energy(h: TensorVariable) -> TensorVariable:
        return aet.sum(g * h)

    def kinetic_energy(p: TensorVariable) -> TensorVariable:
        v = inverse_mass_matrix * p
        return aet.sum(0.5 * aet.dot(v, p))

    return potential_energy, kinetic_energy


def CircularMotion(inverse_mass_matrix):
    def potential_energy(q: TensorVariable) -> TensorVariable:
        return -1.0 / aet.power(aet.square(q[0]) + aet.square(q[1]), 0.5)

    def kinetic_energy(p: TensorVariable) -> TensorVariable:
        return 0.5 * aet.dot(inverse_mass_matrix, aet.square(p))

    return potential_energy, kinetic_energy


integration_examples = [
    {
        "model": FreeFall,
        "n_steps": 100,
        "step_size": 0.01,
        "q_init": np.array([0.0]),
        "p_init": np.array([1.0]),
        "q_final": np.array([0.5]),
        "p_final": np.array([0.0]),
        "inverse_mass_matrix": np.array([1.0]),
    },
    {
        "model": HarmonicOscillator,
        "n_steps": 100,
        "step_size": 0.01,
        "q_init": np.array([0.0]),
        "p_init": np.array([1.0]),
        "q_final": np.array([np.sin(1.0)]),
        "p_final": np.array([np.cos(1.0)]),
        "inverse_mass_matrix": np.array([1.0]),
    },
    {
        "model": CircularMotion,
        "n_steps": 628,
        "step_size": 0.01,
        "q_init": np.array([1.0, 0.0]),
        "p_init": np.array([0.0, 1.0]),
        "q_final": np.array([1.0, 0.0]),
        "p_final": np.array([0.0, 1.0]),
        "inverse_mass_matrix": np.array([1.0, 1.0]),
    },
]


def create_integrate_fn(potential, step_fn, n_steps):
    q = aet.vector("q")
    p = aet.vector("p")
    step_size = aet.scalar("step_size")
    energy = potential(q)
    energy_grad = aesara.grad(energy, q)
    trajectory, _ = aesara.scan(
        fn=step_fn,
        outputs_info=[
            {"initial": q},
            {"initial": p},
            {"initial": energy},
            {"initial": energy_grad},
        ],
        non_sequences=[step_size],
        n_steps=n_steps,
    )
    integrate_fn = aesara.function((q, p, step_size), trajectory)
    return integrate_fn


@pytest.mark.parametrize("example", integration_examples)
def test_velocity_verlet(example):
    model = example["model"]
    inverse_mass_matrix = example["inverse_mass_matrix"]
    step_size = example["step_size"]
    q_init = example["q_init"]
    p_init = example["p_init"]

    potential, kinetic_energy = model(inverse_mass_matrix)
    step = velocity_verlet(potential, kinetic_energy)

    q = aet.vector("q")
    p = aet.vector("p")
    p_final = aet.vector("p_final")
    energy_at = potential(q) + kinetic_energy(p)
    energy_fn = aesara.function((q, p), energy_at)

    integrate_fn = create_integrate_fn(potential, step, example["n_steps"])
    q_final, p_final, energy_final, _ = integrate_fn(q_init, p_init, step_size)

    # Check that the trajectory was correctly integrated
    np.testing.assert_allclose(example["q_final"], q_final[-1], atol=1e-2)
    np.testing.assert_allclose(example["p_final"], p_final[-1], atol=1e-2)

    # Symplectic integrators conserve energy
    energy = energy_fn(q_init, p_init)
    new_energy = energy_fn(q_final[-1], p_final[-1])
    assert energy == pytest.approx(new_energy, 1e-4)
