import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream
from numpy.testing import assert_allclose

from aehmc import mass_matrix


@pytest.mark.parametrize("is_full_matrix", [True, False])
@pytest.mark.parametrize("n_dims", [0, 1, 3])
def test_mass_matrix_adaptation(is_full_matrix, n_dims):

    srng = RandomStream(seed=0)

    if n_dims > 0:
        mu = 0.5 * at.ones((n_dims,))
        cov = 0.33 * at.ones((n_dims, n_dims))
    else:
        mu = at.constant(0.5)
        cov = at.constant(0.33)

    init, update, final = mass_matrix.covariance_adaptation(is_full_matrix)
    _, wc_state = init(n_dims)
    if n_dims > 0:
        dist = srng.multivariate_normal
    else:
        dist = srng.normal

    def one_step(*wc_state):
        sample = dist(mu, cov)
        wc_state = update(sample, wc_state)
        return wc_state

    results, updates = aesara.scan(
        fn=one_step,
        outputs_info=[
            {"initial": wc_state[0]},
            {"initial": wc_state[1]},
            {"initial": wc_state[2]},
        ],
        n_steps=1_000,
    )

    inverse_mass_matrix = final((results[0][-1], results[1][-1], results[2][-1]))

    if n_dims > 0:
        if is_full_matrix:
            expected = cov.eval()
            inverse_mass_matrix = inverse_mass_matrix.eval()
        else:
            expected = np.diagonal(cov.eval())
            inverse_mass_matrix = inverse_mass_matrix.eval()
        assert np.shape(inverse_mass_matrix) == np.shape(expected)
        assert_allclose(inverse_mass_matrix, expected, rtol=0.1)
    else:
        sigma = at.sqrt(inverse_mass_matrix).eval()
        expected_sigma = cov.eval()
        assert np.ndim(expected_sigma) == 0
        assert sigma == pytest.approx(expected_sigma, rel=0.1)
