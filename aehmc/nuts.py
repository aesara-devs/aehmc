from aehmc.termination import iterative_uturn
from aehmc.trajectory import dynamic_integration, multiplicative_expansion


def kernel(
    srng: RandomStream,
    potential_fn: Callable[[TensorVariable], TensorVariable],
    step_size: TensorVariable,
    inverse_mass_matrix: TensorVariable,
    divergence_threshold: int = 1000,
):

    momentum_generator, kinetic_ernergy_fn, uturn_check_fn = metrics.gaussian_metric(
        inverse_mass_matrix
    )
    symplectic_integrator = integrators.velocity_verlet(
        potential_fn, kinetic_ernergy_fn
    )
    new_termination_state, update_termination_state, is_criterion_met = iterative_uturn(
        uturn_check_fn
    )
    trajectory_integrator = dynamic_integration(
        symplectic_integrator,
        kinetic_ernergy_fn,
        update_termination_state,
        is_criterion_met,
        divergence_threshold,
    )
    expand = multiplicative_expansion(
        trajectory_integrator,
        uturn_check_fn,
        step_size,
        max_num_expansions,
    )

    def step(
        q: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
    ):
        p = momentum_generator(srng)
        initial_state = (q, p, potential_energy, potential_energy_grad)
        initial_termination_state = new_termination_state(q, max_num_expansions)
        initial_energy = potential_energy + kinetic_ernergy_fn(p)
        initial_proposal = (
            initial_state,
            initial_energy,
            0.0,
            -np.inf,
        )
        q_new, p_new, potential_energy_new, potential_energy_grad_new = expand(
            srng,
            initial_proposal,
            initial_state,
            initial_state,
            p,
            initial_termination_state,
            initial_energy,
        )
        return q_new, p_new, potential_energy_new, potential_energy_grad_new

    return step
