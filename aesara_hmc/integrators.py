from typing import Callable, Tuple

import aesara
from aesara.tensor.var import TensorVariable

IntegratorStateType = Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable
]


def velocity_verlet(
    potential_fn: Callable[[TensorVariable], TensorVariable],
    kinetic_energy_fn: Callable[[TensorVariable], TensorVariable],
) -> Callable[[IntegratorStateType, TensorVariable], IntegratorStateType]:

    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    def one_step(
        state: IntegratorStateType,
        step_size: TensorVariable,
    ) -> IntegratorStateType:
        position, momentum, potential_energy, potential_energy_grad = state

        new_momentum = momentum - b1 * step_size * potential_energy_grad

        kinetic_grad = aesara.grad(kinetic_energy_fn(new_momentum), [new_momentum])
        new_position = position + a2 * step_size * kinetic_grad

        new_potential_energy = potential_fn(new_position)
        new_potential_energy_grad = aesara.grad(new_potential_energy, [new_position])
        new_momentum = new_momentum - b1 * step_size * new_potential_energy_grad

        return (
            new_position,
            new_momentum,
            new_potential_energy,
            new_potential_energy_grad,
        )

    return one_step
