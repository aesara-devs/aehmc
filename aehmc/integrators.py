from typing import Callable, Tuple

import aesara
from aesara.tensor.var import TensorVariable

IntegratorStateType = Tuple[
    TensorVariable, TensorVariable, TensorVariable, TensorVariable
]


def velocity_verlet(
    potential_fn: Callable[[TensorVariable], TensorVariable],
    kinetic_energy_fn: Callable[[TensorVariable], TensorVariable],
) -> Callable[
    [TensorVariable, TensorVariable, TensorVariable, TensorVariable, TensorVariable],
    IntegratorStateType,
]:
    """The velocity Verlet (or Verlet-Störmer) integrator.

    The velocity Verlet is a two-stage palindromic integrator [1]_ of the form
    (a1, b1, a2, b1, a1) with a1 = 0. It is numerically stable for values of
    the step size that range between 0 and 2 (when the mass matrix is the
    identity).

    References
    ----------
    .. [1]: Bou-Rabee, Nawaf, and Jesús Marıa Sanz-Serna. "Geometric
            integrators and the Hamiltonian Monte Carlo method." Acta Numerica 27
            (2018): 113-206.

    """
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    def one_step(
        position: TensorVariable,
        momentum: TensorVariable,
        potential_energy: TensorVariable,
        potential_energy_grad: TensorVariable,
        step_size: TensorVariable,
    ) -> IntegratorStateType:

        new_momentum = momentum - b1 * step_size * potential_energy_grad

        kinetic_grad = aesara.grad(kinetic_energy_fn(new_momentum), new_momentum)
        new_position = position + a2 * step_size * kinetic_grad

        new_potential_energy = potential_fn(new_position)
        new_potential_energy_grad = aesara.grad(new_potential_energy, new_position)
        new_momentum = new_momentum - b1 * step_size * new_potential_energy_grad

        return (
            new_position,
            new_momentum,
            new_potential_energy,
            new_potential_energy_grad,
        )

    return one_step
