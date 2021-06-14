from typing import Callable, Tuple

import aesara
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.var import TensorVariable
from aesara.scan.utils import until

from aehmc.proposals import proposal_generator, progressive_uniform_sampling

from aehmc.proposals import generate_proposal, progressive_uniform_sampling

IntegratorStateType = Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable
]

ProposalType = Tuple[
    IntegratorStateType, TensorVariable, TensorVariable, TensorVariable
]

TerminationStateType = Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable
]

TrajectoryType = Tuple[
    IntegratorStateType, IntegratorStateType, TensorVariable, TensorVariable
]

<<<<<<< HEAD
=======

def append_to_trajectory(trajectory, new_state):
    """Append a state to the right of a trajectory."""
    return (trajectory[0], new_state, trajectory[2] + new_state[1], trajectory[3] + 1)


>>>>>>> 49673bd (draft dynamic integration)
# -------------------------------------------------------------------
#                       STATIC INTEGRATION
#
# This section contains algorithms that integrate the trajectory for
# a set number of integrator steps.
# -------------------------------------------------------------------


def static_integration(
    integrator: Callable,
    step_size: float,
    num_integration_steps: int,
    direction: int = 1,
) -> Callable:
    """Generate a trajectory by integrating several times in one direction."""

    directed_step_size = direction * step_size

    def integrate(q_init, p_init, energy_init, energy_grad_init) -> IntegratorStateType:
        def one_step(q, p, energy, energy_grad):
            new_state = integrator(q, p, energy, energy_grad, directed_step_size)
            return new_state

        [q, p, energy, energy_grad], _ = aesara.scan(
            fn=one_step,
            outputs_info=[
                {"initial": q_init},
                {"initial": p_init},
                {"initial": energy_init},
                {"initial": energy_grad_init},
            ],
            n_steps=num_integration_steps,
        )

        return q[-1], p[-1], energy[-1], energy_grad[-1]

    return integrate


# -------------------------------------------------------------------
#                       DYNAMIC INTEGRATION
#
# This section contains algorithms that determine the number of
# integrator steps dynamically using a termination criterion that
# is updated at every step.
# -------------------------------------------------------------------


def dynamic_integration(
    integrator: Callable,
    kinetic_energy: Callable,
    update_termination_state: Callable,
    is_criterion_met: Callable,
    divergence_threshold: TensorVariable,
):
    """Integrate a trajectory and update the proposal sequentially in one direction
    until the termination criterion is met.

    Parameters
    ----------
    integrator
        The symplectic integrator used to integrate the hamiltonian trajectory.
    kinetic_energy
        Function to compute the current value of the kinetic energy.
    update_termination_state
        Updates the state of the termination mechanism.
    is_criterion_met
        Determines whether the termination criterion has been met.
    divergence_threshold
        Value of the difference of energy between two consecutive states above which we say a transition is divergent.

    """
    _, generate_proposal = proposal_generator(kinetic_energy, divergence_threshold)
    sample_proposal = progressive_uniform_sampling

    def integrate(
        srng: RandomStream,
        previous_state: IntegratorStateType,
        direction: TensorVariable,
        termination_state: TerminationStateType,
        max_num_steps: TensorVariable,
        step_size: TensorVariable,
        initial_energy: TensorVariable,
    ):
        """Integrate the trajectory starting from `initial_state` and update
        the proposal sequentially until the termination criterion is met.

        Parameters
        ----------
        rng_key
            Key used by JAX's random number generator.
        previous_state
            The last state of the previously integrated trajectory.
        direction int in {-1, 1}
            The direction in which to expand the trajectory.
        termination_state
            The state that keeps track of the information needed for the termination criterion.
        max_num_steps
            The maximum number of integration steps. The expansion will stop
            when this number is reached if the termination criterion has not
            been met.
        step_size
            The step size of the symplectic integrator.
        initial_energy
            Initial energy H0 of the HMC step (not to confused with the initial energy of the subtree)

        """

        def take_first_step(
            previous_last_state: IntegratorStateType,
            termination_state: TerminationStateType,
        ):
            """The first state of the new trajectory is obtained by integrating
            once starting from the last state of the previous trajectory.

            """
            initial_state = integrator(*previous_last_state, direction * step_size)
            initial_proposal, _ = generate_proposal(initial_energy, initial_state)
            initial_trajectory = (initial_state, initial_state, initial_state[1], 1)
            initial_termination_state = update_termination_state(
                termination_state, initial_trajectory[2], initial_state[1]
            )
            return (
                initial_proposal,
                initial_trajectory,
                initial_termination_state,
            )

        def do_keep_integrating(is_diverging, has_terminated):
            return ~has_terminated & ~is_diverging

        def add_one_state(
            proposal: ProposalType,
            trajectory: TrajectoryType,
            termination_state: TerminationStateType,
        ):
            last_state = trajectory[1]
            new_state = integrator(*last_state, direction * step_size)
            new_proposal, is_diverging = generate_proposal(initial_energy, new_state)

            new_trajectory = append_to_trajectory(trajectory, new_state)
            sampled_proposal = sample_proposal(srng, proposal, new_proposal)

            momentum_sum = new_trajectory[3]
            momentum = new_state[2]
            new_termination_state = update_termination_state(
                termination_state, momentum_sum, momentum
            )
            has_terminated = is_criterion_met(
                new_termination_state, momentum_sum, momentum
            )

            return (sampled_proposal, new_trajectory, new_termination_state), until(
                ~do_keep_integrating(is_diverging, has_terminated)
            )

        proposal, trajectory, termination_state = take_first_step(previous_state, termination_state)

        _ = aesara.scan(
            add_one_state,
            outputs_info=(proposal, trajectory, termination_state),
            num_steps=max_num_steps,
        )

        return _

    return integrate


def multiplicative_expansion():
    raise NotImplementedError
