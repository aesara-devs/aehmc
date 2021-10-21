import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream
from numpy.testing import assert_allclose

from aehmc import mass_matrix


@pytest.mark.parametrize("is_full_matrix", [True, False])
@pytest.mark.parametrize("n_dims", [1, 3])
def test_mass_matrix_adaptation(is_full_matrix, n_dims):

    srng = RandomStream(seed=0)
    mu = 0.5 * at.ones((n_dims,))
    cov = 0.33 * at.ones((n_dims, n_dims))

    init, update, final = mass_matrix.covariance_adaptation(is_full_matrix)
    _, wc_state = init(n_dims)

    def one_step(*wc_state):
        sample = srng.multivariate_normal(mu, cov)
        wc_state = update(sample, wc_state)
        return wc_state

    results, updates = aesara.scan(
        fn=one_step,
        outputs_info=[
            {"initial": wc_state[0]},
            {"initial": wc_state[1]},
            {"initial": wc_state[2]},
        ],
        n_steps=2000,
    )

    inverse_mass_matrix = final((results[0][-1], results[1][-1], results[2][-1]))

    if is_full_matrix:
        assert_allclose(inverse_mass_matrix.eval(), cov.eval(), rtol=0.1)
    else:
        diag_cov = np.diagonal(cov.eval())
        assert_allclose(inverse_mass_matrix.eval(), diag_cov, rtol=0.1)
