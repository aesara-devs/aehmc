import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.var import TensorVariable

from aesara_hmc.integrators import velocity_verlet


def HarmonicOscillator(inverse_mass_matrix, k=5, m=1.0):
    """Potential and Kinetic energy of an harmonic oscillator."""

    def potential_energy(x: TensorVariable) -> TensorVariable:
        return aet.sum(0.5 * k * aet.square(x))

    def kinetic_energy(p: TensorVariable) -> TensorVariable:
        v = inverse_mass_matrix * p
        return aet.sum(0.5 * aet.dot(v, p))

    return potential_energy, kinetic_energy


def FreeFall(inverse_mass_matrix, g=9.81, m=1.0):
    """Potential and kinetic energy of a free-falling object."""

    def potential_energy(h: TensorVariable) -> TensorVariable:
        return aet.sum(m * g * h)

    def kinetic_energy(p: TensorVariable) -> TensorVariable:
        v = inverse_mass_matrix * p
        return aet.sum(0.5 * aet.dot(v, p))

    return potential_energy, kinetic_energy


integration_examples = [
    {
        "model": HarmonicOscillator,
        "n_steps": 100,
        "step_size": 0.01,
        "q_init": np.array([0.0]),
        "p_init": np.array([1.0]),
        "inverse_mass_matrix": np.array([1.0]),
    },
    {
        "model": FreeFall,
        "n_steps": 100,
        "step_size": 0.01,
        "q_init": np.array([0.0]),
        "p_init": np.array([1.0]),
        "inverse_mass_matrix": np.array([1.0]),
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
    """The import characteristic of symplectic integrators is the conservation of energy."""

    model = example["model"]
    inverse_mass_matrix = example["inverse_mass_matrix"]
    step_size = example["step_size"]
    q_init = example["q_init"]
    p_init = example["p_init"]

    potential, kinetic_energy = model(inverse_mass_matrix)
    step = velocity_verlet(potential, kinetic_energy)

    q = aet.vector("q")
    p = aet.vector("p")
    # XXX: Should this be a matrix/column vector?
    p_final = aet.vector("p_final")
    energy_at = potential(q) + kinetic_energy(p)
    energy_new_energy_fn = aesara.function(
        (q, p, p_final), (energy_at, energy_at + kinetic_energy(p_final))
    )

    integrate_fn = create_integrate_fn(potential, step, example["n_steps"])
    q_final, p_final, energy_final, _ = integrate_fn(q_init, p_init, step_size)

    # Symplectic integrators conserve energy
    energy, new_energy = energy_new_energy_fn(q_init, p_init, p_final.squeeze())
    assert energy == pytest.approx(new_energy, 1e-3)
