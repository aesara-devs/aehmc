from typing import Callable, Tuple

import aesara.tensor as at
from aesara import config
from aesara.tensor.var import TensorVariable

from aehmc import algorithms

WelfordAlgorithmState = Tuple[TensorVariable, TensorVariable, TensorVariable]


def covariance_adaptation(
    is_mass_matrix_full: bool = False,
) -> Tuple[Callable, Callable, Callable]:
    """Adapts the values in the mass matrix by computing the covariance
    between parameters.

    Parameters
    ----------
    is_mass_matrix_full
        When False the algorithm adapts and returns a diagonal mass matrix
        (default), otherwise adapts and returns a dense mass matrix.

    Returns
    -------
    init
        A function that initializes the step of the mass matrix adaptation.
    update
        A function that updates the state of the mass matrix.
    final
        A function that computes the inverse mass matrix based on the current
        state.
    """

    wc_init, wc_update, wc_final = algorithms.welford_covariance(is_mass_matrix_full)

    def init(
        n_dims: int,
    ) -> Tuple[TensorVariable, WelfordAlgorithmState]:
        """Initialize the mass matrix adaptation.

        Parameters
        ----------
        ndims
            The number of dimensions of the mass matrix, which corresponds to
            the number of dimensions of the chain position.

        Returns
        -------
        The initial value of the mass matrix and the initial state of the
        Welford covariance algorithm.

        """
        if n_dims == 0:
            inverse_mass_matrix = at.constant(1.0, dtype=config.floatX)
        elif is_mass_matrix_full:
            inverse_mass_matrix = at.eye(n_dims, dtype=config.floatX)
        else:
            inverse_mass_matrix = at.ones((n_dims,), dtype=config.floatX)

        wc_state = wc_init(n_dims)

        return inverse_mass_matrix, wc_state

    def update(
        position: TensorVariable, wc_state: WelfordAlgorithmState
    ) -> WelfordAlgorithmState:
        """Update the algorithm's state.

        Parameters
        ----------
        position
            The current position of the chain.
        wc_state
            Current state of Welford's algorithm to compute covariance.

        """
        new_wc_state = wc_update(position, *wc_state)
        return new_wc_state

    def final(wc_state: WelfordAlgorithmState) -> TensorVariable:
        """Final iteration of the mass matrix adaptation.

        In this step we compute the mass matrix from the covariance matrix computed
        by the Welford algorithm, applying the shrinkage used in Stan [1]_.

        Parameters
        ----------
        wc_state
            Current state of Welford's algorithm to compute covariance.

        Returns
        -------
        The value of the inverse mass matrix computed from the covariance estimate.

        References
        ----------
        .. [1]: Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B.,
                Betancourt, M., ... & Riddell, A. (2017). Stan: A probabilistic programming
                language. Journal of statistical software, 76(1), 1-32.

        """
        _, m2, sample_size = wc_state
        covariance = wc_final(m2, sample_size)

        scaled_covariance = (sample_size / (sample_size + 5)) * covariance
        shrinkage = 1e-3 * (5 / (sample_size + 5))
        if covariance.ndim > 0:
            if is_mass_matrix_full:
                new_inverse_mass_matrix = (
                    scaled_covariance + shrinkage * at.identity_like(covariance)
                )
            else:
                new_inverse_mass_matrix = scaled_covariance + shrinkage
        else:
            new_inverse_mass_matrix = scaled_covariance + shrinkage

        return new_inverse_mass_matrix

    return init, update, final
