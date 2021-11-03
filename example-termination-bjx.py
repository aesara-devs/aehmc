import jax
import jax.numpy as jnp

from blackjax.inference.hmc import metrics, trajectory, integrators, termination

rng_key = jax.random.PRNGKey(0)


def potential_fn(x):
    return -jax.scipy.stats.norm.logpdf(x)


step_size = 1e-6
position = 1.0
inverse_mass_matrix = jnp.array([1.0])

momentum_generator, kinetic_energy_fn, uturn_check_fn = metrics.gaussian_euclidean(
    inverse_mass_matrix
)

integrator = integrators.velocity_verlet(potential_fn, kinetic_energy_fn)
(
    new_criterion_state,
    update_criterion_state,
    is_criterion_met,
) = termination.iterative_uturn_numpyro(uturn_check_fn)

trajectory_integrator = trajectory.dynamic_progressive_integration(
    integrator,
    kinetic_energy_fn,
    update_criterion_state,
    is_criterion_met,
    1000,
)

# Initialize
direction = 1
initial_state = integrators.new_integrator_state(
    potential_fn, position, momentum_generator(rng_key, position)
)
initial_energy = initial_state.potential_energy + kinetic_energy_fn(
    initial_state.momentum
)
termination_state = new_criterion_state(initial_state, 10)


def body_fn(step, momentum_sum, state, termination_state):
    state = integrator(state, 1e-2)
    termination_state = update_criterion_state(termination_state, momentum_sum, state.momentum, step)
    is_turning = is_criterion_met(termination_state, momentum_sum, state.momentum)
