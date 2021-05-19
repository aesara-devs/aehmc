from typing import Callable, Tuple

import aesara.tensor as aet
import numpy as np
import scipy.linalg
import scipy.stats
from aesara.tensor.var import TensorVariable


def gaussian_metric(
    inverse_mass_matrix: TensorVariable,
) -> Tuple[Callable, Callable, Callable]:
    shape = np.shape(inverse_mass_matrix)[0]

    if inverse_mass_matrix.ndim == 1:
        mass_matrix_sqrt = np.sqrt(np.reciprocal(inverse_mass_matrix))
        dot, matmul = lambda x, y: x * y, lambda x, y: x * y
    elif inverse_mass_matrix.ndim == 2:
        tril_inv = scipy.linalg.cholesky(inverse_mass_matrix)
        identity = np.identity(shape)
        mass_matrix_sqrt = scipy.linalg.solve_triangular(tril_inv, identity, lower=True)
        dot, matmul = aet.dot, aet.dot
    else:
        raise ValueError(
            f"Expected a mass matrix of dimension 1 (diagonal) or 2, got {inverse_mass_matrix.ndim}"
        )

    def momentum_generator(srng):
        norm_samples = srng.normal(0, 1, size=shape)
        momentum = dot(norm_samples, mass_matrix_sqrt)
        return momentum

    def kinetic_energy(momentum: TensorVariable):
        velocity = matmul(inverse_mass_matrix, momentum)
        kinetic_energy = 0.5 * aet.dot(velocity, momentum)
        return kinetic_energy

    def is_turning(
        momentum_left: TensorVariable,
        momentum_right: TensorVariable,
        momentum_sum: TensorVariable,
    ):
        velocity_left = matmul(inverse_mass_matrix, momentum_left)
        velocity_right = matmul(inverse_mass_matrix, momentum_right)

        rho = momentum_sum - (momentum_right + momentum_left) / 2
        turning_at_left = aet.dot(velocity_left, rho) <= 0
        turning_at_right = aet.dot(velocity_right, rho) <= 0

        return turning_at_left | turning_at_right

    return momentum_generator, kinetic_energy, is_turning
