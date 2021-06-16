import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aeppl.logprob import logprob
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc.integrators import new_integrator_state, velocity_verlet
from aehmc.metrics import gaussian_metric
from aehmc.termination import iterative_uturn
from aehmc.trajectory import dynamic_integration, static_integration


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


@pytest.mark.parametrize("case", [(0.0001, False), (1000, True)])
def test_dynamic_integration_divergence(case):
    srng = RandomStream(seed=59)

    def potential_fn(x):
        return -aet.sum(logprob(aet.random.normal(0.0, 1.0), x))

    should_diverge = case[1]

    # Set up the trajectory integrator
    inverse_mass_matrix = aet.ones(1)

    momentum_generator, kinetic_energy_fn, uturn_check_fn = gaussian_metric(
        inverse_mass_matrix
    )
    integrator = velocity_verlet(potential_fn, kinetic_energy_fn)
    (
        new_criterion_state,
        update_criterion_state,
        is_criterion_met,
    ) = iterative_uturn(uturn_check_fn)

    trajectory_integrator = dynamic_integration(
        integrator,
        kinetic_energy_fn,
        update_criterion_state,
        is_criterion_met,
        divergence_threshold=aet.as_tensor(1000),
    )

    # Initialize the state
    direction = aet.as_tensor(1)
    step_size = aet.as_tensor(case[0])
    max_num_steps = aet.as_tensor(100)
    num_doublings = aet.as_tensor(10)
    position = aet.as_tensor(np.ones(1))

    initial_state = new_integrator_state(
        potential_fn, position, momentum_generator(srng)
    )
    initial_energy = initial_state[2] + kinetic_energy_fn(initial_state[1])
    termination_state = new_criterion_state(initial_state[0], num_doublings)

    state, updates = trajectory_integrator(
        srng,
        initial_state,
        direction,
        termination_state,
        max_num_steps,
        step_size,
        initial_energy,
    )

    state_fn = aesara.function((), state, updates=updates, on_unused_input="ignore")

    is_diverging = state_fn()[-2]

    assert is_diverging.item() is should_diverge
