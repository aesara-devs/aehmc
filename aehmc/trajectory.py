from typing import Callable, Tuple

import aesara
from aesara.tensor.var import TensorVariable

IntegratorStateType = Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable
]

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


def dynamic_integration():
    raise NotImplementedError


def multiplicative_expansion():
    raise NotImplementedError
