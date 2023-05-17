from typing import Callable, NamedTuple

import aesara
from aesara.tensor.var import TensorVariable


class IntegratorState(NamedTuple):
    position: TensorVariable
    momentum: TensorVariable
    potential_energy: TensorVariable
    potential_energy_grad: TensorVariable


def new_integrator_state(
    potential_fn: Callable, position: TensorVariable, momentum: TensorVariable
) -> IntegratorState:
    """Create a new integrator state from the current values of the position and momentum."""
    potential_energy = potential_fn(position)
    return IntegratorState(
        position=position,
        momentum=momentum,
        potential_energy=potential_energy,
        potential_energy_grad=aesara.grad(potential_energy, position),
    )


def velocity_verlet(
    potential_fn: Callable[[TensorVariable], TensorVariable],
    kinetic_energy_fn: Callable[[TensorVariable], TensorVariable],
) -> Callable[[IntegratorState, TensorVariable], IntegratorState]:
    """The velocity Verlet (or Verlet-Störmer) integrator.

    The velocity Verlet is a two-stage palindromic integrator [1]_ of the form
    (a1, b1, a2, b1, a1) with a1 = 0. It is numerically stable for values of
    the step size that range between 0 and 2 (when the mass matrix is the
    identity).

    Parameters
    ----------
    potential_fn
        A function that returns the potential energy of a chain at a given
        position.
    kinetic_energy_fn
        A function that returns the kinetic energy of a chain at a given
        position and a given momentum.

    References
    ----------
    .. [1]: Bou-Rabee, Nawaf, and Jesús Marıa Sanz-Serna. "Geometric
            integrators and the Hamiltonian Monte Carlo method." Acta Numerica 27
            (2018): 113-206.

    """
    a1 = 0
    b1 = 0.5
    a2 = 1 - 2 * a1

    def one_step(state: IntegratorState, step_size: TensorVariable) -> IntegratorState:
        momentum = state.momentum - b1 * step_size * state.potential_energy_grad

        kinetic_grad = aesara.grad(kinetic_energy_fn(momentum), momentum)
        position = state.position + a2 * step_size * kinetic_grad

        potential_energy = potential_fn(position)
        potential_energy_grad = aesara.grad(potential_energy, position)
        momentum = momentum - b1 * step_size * potential_energy_grad

        return IntegratorState(
            position=position,
            momentum=momentum,
            potential_energy=potential_energy,
            potential_energy_grad=potential_energy_grad,
        )

    return one_step
