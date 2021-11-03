import aesara
import aesara.tensor as at
from aesara.tensor.random.utils import RandomStream
import numpy as np

from aehmc.metrics import gaussian_metric
from aehmc.termination import iterative_uturn
from aehmc.integrators import new_integrator_state, velocity_verlet
from aeppl.logprob import logprob


srng = RandomStream(0)


def potential_fn(x):
    return -at.sum(logprob(at.random.normal(0.0, 1.0), x))


inverse_mass_matrix = at.ones(1)
direction = at.as_tensor(1)
step_size = at.as_tensor(1e-6)
position = at.ones(1)


momentum_generator_fn, kinetic_energy_fn, is_turning_fn = gaussian_metric(inverse_mass_matrix)
(
    new_criterion_state,
    update_criterion_state,
    is_criterion_met_fn,
) = iterative_uturn(is_turning_fn)
integrator = velocity_verlet(potential_fn, kinetic_energy_fn)


# Initialize the state
momentum = momentum_generator_fn(srng)
state = new_integrator_state(potential_fn, position, momentum)
energy = state[2] + kinetic_energy_fn(state[1])
proposal = (
    state,
    energy,
    at.as_tensor(0.0, dtype="float64"),
    at.as_tensor(-np.inf, dtype="float64"),
)
termination_state = new_criterion_state(state[0], 10)

# Take one step
state = integrator(*state, 1e-6)
momentum_sum = state[1]
termination_state = update_criterion_state(termination_state, momentum_sum, state[1], 0)

def step_fn(step, momentum_sum, q, p, pe, peg, mc, msc, imin, imax):
    state = (q, p, pe, peg)
    termination_state = (mc, msc, imin, imax)

    state = integrator(*state, 1e-2)
    momentum_sum = momentum_sum + state[1]
    termination_state = update_criterion_state(termination_state, momentum_sum, state[1], step)
    is_criterion_met = is_criterion_met_fn(termination_state, momentum_sum, state[1])

    return (momentum_sum,) + state + termination_state + (is_criterion_met, step)

steps = at.arange(1, 10)
results, update = aesara.scan(
        step_fn,
        outputs_info=[momentum_sum, *state, *termination_state, None, None],
        sequences=steps,
)

fn = aesara.function((), results, updates=update)
print(fn())
