import aesara
import aesara.tensor as at
import numpy as np
import pytest
from aesara import config

from aehmc import algorithms


def test_dual_averaging():
    """Find the minimum of a simple function using Dual Averaging."""

    def fn(x):
        return (x - 1) ** 2

    init, update = algorithms.dual_averaging(gamma=0.5)

    def one_step(step, x, x_avg, gradient_avg, mu):
        value = fn(x)
        gradient = aesara.grad(value, x)
        return update(gradient, step, x, x_avg, gradient_avg, mu)

    mu = at.as_tensor(at.constant(0.5), dtype=config.floatX)
    step, x_init, x_avg, gradient_avg, mu = init(mu)

    states, updates = aesara.scan(
        fn=one_step,
        outputs_info=[
            {"initial": step},
            {"initial": x_init},
            {"initial": x_avg},
            {"initial": gradient_avg},
            {"initial": mu},
        ],
        n_steps=100,
    )

    last_x = states[1].eval()[-1]
    last_x_avg = states[2].eval()[-1]
    assert last_x_avg == pytest.approx(1.0, 1e-2)
    assert last_x == pytest.approx(1.0, 1e-2)


@pytest.mark.parametrize("num_dims", [0, 1, 3])
@pytest.mark.parametrize("do_compute_covariance", [True, False])
def test_welford_constant(num_dims, do_compute_covariance):
    num_samples = 10

    if num_dims > 0:
        sample = at.ones((num_dims,))  # constant samples
    else:
        sample = at.constant(1.0)

    init, update, final = algorithms.welford_covariance(do_compute_covariance)
    state = init(num_dims)
    for _ in range(num_samples):
        state = update(sample, *state)

    mean = state[0].eval()
    if num_dims > 0:
        expected = np.ones(num_dims)
        assert np.shape(mean) == np.shape(expected)
        np.testing.assert_allclose(mean, expected, rtol=1e-1)
    else:
        assert np.ndim(mean) == 0
        assert mean == 1.0

    cov = final(state[1], state[2]).eval()
    if num_dims > 0:
        if do_compute_covariance:
            expected = np.zeros((num_dims, num_dims))
        else:
            expected = np.zeros(num_dims)

        assert np.shape(cov) == np.shape(expected)
        np.testing.assert_allclose(cov, expected)
    else:
        assert np.ndim(cov) == 0
        assert cov == 0


@pytest.mark.parametrize("do_compute_covariance", [True, False])
@pytest.mark.parametrize("n_dim", [1, 3])
def test_welford(n_dim, do_compute_covariance):
    num_samples = 10

    init, update, final = algorithms.welford_covariance(do_compute_covariance)
    state = init(n_dim)
    for i in range(num_samples):
        sample = i * at.ones(n_dim)
        state = update(sample, *state)

    mean = state[0].eval()
    expected = (9.0 / 2) * np.ones(n_dim)
    np.testing.assert_allclose(mean, expected)

    cov = final(state[1], state[2]).eval()
    if do_compute_covariance:
        expected = 55.0 / 6.0 * np.ones((n_dim, n_dim))
    else:
        expected = 55.0 / 6.0 * np.ones(n_dim)
    assert np.shape(cov) == np.shape(expected)
    np.testing.assert_allclose(cov, expected)


@pytest.mark.parametrize("do_compute_covariance", [True, False])
def test_welford_scalar(do_compute_covariance):
    """ "Test the Welford algorithm when the state is a scalar."""
    num_samples = 10

    init, update, final = algorithms.welford_covariance(do_compute_covariance)
    state = init(0)
    for i in range(num_samples):
        sample = at.as_tensor(i)
        state = update(sample, *state)

    cov = final(state[1], state[2]).eval()
    assert np.ndim(cov) == 0
    assert pytest.approx(cov.squeeze()) == 55.0 / 6.0
