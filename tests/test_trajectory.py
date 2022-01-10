import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aeppl.logprob import logprob
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable

from aehmc.integrators import new_integrator_state, velocity_verlet
from aehmc.metrics import gaussian_metric
from aehmc.termination import iterative_uturn
from aehmc.trajectory import (
    dynamic_integration,
    multiplicative_expansion,
    static_integration,
)

aesara.config.optimizer = "fast_compile"
aesara.config.exception_verbosity = "high"


def CircularMotion(inverse_mass_matrix):
    def potential_energy(q: TensorVariable) -> TensorVariable:
        return -1.0 / at.power(at.square(q[0]) + at.square(q[1]), 0.5)

    def kinetic_energy(p: TensorVariable) -> TensorVariable:
        return 0.5 * at.dot(inverse_mass_matrix, at.square(p))

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
    integrator = static_integration(step, num_steps)

    q = at.vector("q")
    p = at.vector("p")
    energy = potential(q)
    energy_grad = aesara.grad(energy, q)
    final_state = integrator(q, p, energy, energy_grad, step_size)
    integrate_fn = aesara.function((q, p), final_state)

    q_final, p_final, *_ = integrate_fn(q_init, p_init)

    np.testing.assert_allclose(q_final, example["q_final"], atol=1e-1)
    np.testing.assert_allclose(p_final, example["p_final"], atol=1e-1)


@pytest.mark.parametrize(
    "case",
    [
        (0.0000001, False, False),
        (1000, True, True),
    ],
)
def test_dynamic_integration(case):
    srng = RandomStream(seed=59)

    def potential_fn(x):
        return -at.sum(logprob(at.random.normal(0.0, 1.0), x))

    step_size, should_diverge, should_turn = case

    # Set up the trajectory integrator
    inverse_mass_matrix = at.ones(1)

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
        srng,
        integrator,
        kinetic_energy_fn,
        update_criterion_state,
        is_criterion_met,
        divergence_threshold=at.as_tensor(1000),
    )

    # Initialize the state
    direction = at.as_tensor(1)
    step_size = at.as_tensor(step_size)
    max_num_steps = at.as_tensor(10)
    num_doublings = at.as_tensor(10)
    position = at.as_tensor(np.ones(1))

    initial_state = new_integrator_state(
        potential_fn, position, momentum_generator(srng)
    )
    initial_energy = initial_state[2] + kinetic_energy_fn(initial_state[1])
    termination_state = new_criterion_state(initial_state[0], num_doublings)

    state, updates = trajectory_integrator(
        initial_state,
        direction,
        termination_state,
        max_num_steps,
        step_size,
        initial_energy,
    )

    is_turning = aesara.function((), state[-1], updates=updates)()
    is_diverging = aesara.function((), state[-2], updates=updates)()

    assert is_diverging.item() is should_diverge
    assert is_turning.item() is should_turn


@pytest.mark.parametrize(
    "step_size, should_diverge, should_turn, expected_doublings",
    [
        (100000.0, True, True, 1),
        (0.0000001, False, False, 10),
        (1.0, False, True, 1),
    ],
)
def test_multiplicative_expansion(
    step_size, should_diverge, should_turn, expected_doublings
):
    srng = RandomStream(seed=59)

    def potential_fn(x):
        return 0.5 * at.sum(at.square(x))

    step_size = at.as_tensor(step_size)
    inverse_mass_matrix = at.as_tensor(1.0, dtype="float64")
    position = at.as_tensor(1.0, dtype="float64")

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
        srng,
        integrator,
        kinetic_energy_fn,
        update_criterion_state,
        is_criterion_met,
        divergence_threshold=at.as_tensor(1000),
    )

    expand = multiplicative_expansion(srng, trajectory_integrator, uturn_check_fn, 10)

    # Create the initial state
    state = new_integrator_state(potential_fn, position, momentum_generator(srng))
    energy = state[2] + kinetic_energy_fn(state[1])
    proposal = (
        state,
        energy,
        at.as_tensor(0.0, dtype="float64"),
        at.as_tensor(-np.inf, dtype="float64"),
    )
    termination_state = new_criterion_state(state[0], 10)
    result, updates = expand(
        proposal, state, state, state[1], termination_state, energy, step_size
    )
    fn = aesara.function((), result, updates=updates)
    result = fn()

    num_doublings = result[-3][-1]
    does_diverge = result[-2][-1]
    does_turn = result[-1][-1]

    assert does_diverge == should_diverge
    assert does_turn == should_turn
    assert expected_doublings == num_doublings
