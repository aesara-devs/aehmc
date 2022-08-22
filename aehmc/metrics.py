from typing import Callable, Tuple

import aesara.tensor as at
from aesara.tensor.random.utils import RandomStream
from aesara.tensor.shape import shape_tuple
from aesara.tensor.slinalg import cholesky, solve_triangular
from aesara.tensor.var import TensorVariable


def gaussian_metric(
    inverse_mass_matrix: TensorVariable,
) -> Tuple[Callable, Callable, Callable]:
    r"""Hamiltonian dynamic on euclidean manifold with normally-distributed momentum.

    The gaussian euclidean metric is a euclidean metric further characterized
    by setting the conditional probability density :math:`\pi(momentum|position)`
    to follow a standard gaussian distribution. A Newtonian hamiltonian
    dynamics is assumed.

    Arguments
    ---------
    inverse_mass_matrix
        One or two-dimensional array corresponding respectively to a diagonal
        or dense mass matrix.

    Returns
    -------
    momentum_generator
        A function that generates a value for the momentum at random.
    kinetic_energy
        A function that returns the kinetic energy given the momentum.
    is_turning
        A function that determines whether a trajectory is turning back on
        itself given the values of the momentum along the trajectory.

    References
    ----------
    .. [1]: Betancourt, Michael. "A general metric for Riemannian manifold
            Hamiltonian Monte Carlo." International Conference on Geometric Science of
            Information. Springer, Berlin, Heidelberg, 2013.

    """

    if inverse_mass_matrix.ndim == 0:
        shape: Tuple = ()
        mass_matrix_sqrt = at.sqrt(at.reciprocal(inverse_mass_matrix))
        dot, matmul = lambda x, y: x * y, lambda x, y: x * y
    elif inverse_mass_matrix.ndim == 1:
        shape = (shape_tuple(inverse_mass_matrix)[0],)
        mass_matrix_sqrt = at.sqrt(at.reciprocal(inverse_mass_matrix))
        dot, matmul = at.dot, lambda x, y: x * y
    elif inverse_mass_matrix.ndim == 2:
        shape = (shape_tuple(inverse_mass_matrix)[0],)
        tril_inv = cholesky(inverse_mass_matrix)
        identity = at.eye(*shape)
        mass_matrix_sqrt = solve_triangular(tril_inv, identity, lower=True)
        dot, matmul = at.dot, at.dot
    else:
        raise ValueError(
            f"Expected a mass matrix of dimension 1 (diagonal) or 2, got {inverse_mass_matrix.ndim}"
        )

    def momentum_generator(srng: RandomStream) -> TensorVariable:
        norm_samples = srng.normal(0, 1, size=shape, name="momentum")
        momentum = matmul(mass_matrix_sqrt, norm_samples)
        return momentum

    def kinetic_energy(momentum: TensorVariable) -> TensorVariable:
        velocity = matmul(inverse_mass_matrix, momentum)
        kinetic_energy = 0.5 * dot(velocity, momentum)
        return kinetic_energy

    def is_turning(
        momentum_left: TensorVariable,
        momentum_right: TensorVariable,
        momentum_sum: TensorVariable,
    ) -> bool:
        """Generalized U-turn criterion.

        Parameters
        ----------
        momentum_left
            Momentum of the leftmost point of the trajectory.
        momentum_right
            Momentum of the rightmost point of the trajectory.
        momentum_sum
            Sum of the momenta along the trajectory.

        .. [1]: Betancourt, Michael J. "Generalizing the no-U-turn sampler to Riemannian manifolds." arXiv preprint arXiv:1304.1920 (2013).
        .. [2]: "NUTS misses U-turn, runs in cicles until max depth", Stan Discourse Forum
                https://discourse.mc-stan.org/t/nuts-misses-u-turns-runs-in-circles-until-max-treedepth/9727/46
        """
        velocity_left = matmul(inverse_mass_matrix, momentum_left)
        velocity_right = matmul(inverse_mass_matrix, momentum_right)

        rho = momentum_sum - (momentum_right + momentum_left) / 2
        turning_at_left = at.dot(velocity_left, rho) <= 0
        turning_at_right = at.dot(velocity_right, rho) <= 0

        is_turning = turning_at_left | turning_at_right

        return is_turning

    return momentum_generator, kinetic_energy, is_turning
