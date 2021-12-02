import aesara
import aesara.tensor as at
import arviz
import numpy as np
import pytest
import scipy.stats as stats
from aeppl import joint_logprob
from aesara.tensor.var import TensorVariable

from aehmc import hmc, nuts


def normal_logprob(q: TensorVariable):
    y = (q - 3.0) / 5.0
    return -at.sum(at.square(y))


def test_hmc():
    """Test the HMC kernel on a gaussian target."""
    step_size = 1.0
    inverse_mass_matrix = at.as_tensor(1.0)
    num_integration_steps = 10

    srng = at.random.RandomStream(seed=0)
    Y_rv = srng.normal(1, 2)

    def logprob_fn(y):
        logprob = joint_logprob({Y_rv: y})
        return logprob

    kernel = hmc.kernel(
        srng,
        logprob_fn,
        inverse_mass_matrix,
        num_integration_steps,
    )

    y_vv = Y_rv.clone()
    initial_state = hmc.new_state(y_vv, logprob_fn)

    trajectory, updates = aesara.scan(
        kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            None,
        ],
        non_sequences=step_size,
        n_steps=2_000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory[0], updates=updates)

    samples = trajectory_generator(3.0)
    assert np.mean(samples[1000:]) == pytest.approx(1.0, rel=1e-1)
    assert np.var(samples[1000:]) == pytest.approx(4.0, rel=1e-1)


def test_nuts():
    """Test the NUTS kernel on a gaussian target."""
    step_size = 1.0
    inverse_mass_matrix = at.as_tensor(1.0)

    srng = at.random.RandomStream(seed=0)
    Y_rv = srng.normal(1, 2)

    def logprob_fn(y):
        logprob = joint_logprob({Y_rv: y})
        return logprob

    kernel = nuts.kernel(
        srng,
        logprob_fn,
        inverse_mass_matrix,
    )

    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    trajectory, updates = aesara.scan(
        kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            None,
            None,
            None,
            None,
        ],
        non_sequences=step_size,
        n_steps=2000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory[0], updates=updates)

    samples = trajectory_generator(3.0)
    assert np.mean(samples[1000:]) == pytest.approx(1.0, rel=1e-1)
    assert np.var(samples[1000:]) == pytest.approx(4.0, rel=1e-1)


def assert_mcse(samples, true_param, p_val=0.01):
    d = arviz.convert_to_dataset(np.expand_dims(samples, axis=0))
    ess = np.array(arviz.ess(d).to_array())
    posterior_mean = np.mean(samples, axis=0)
    posterior_sd = np.std(samples, axis=0, ddof=1)
    avg_monte_carlo_standard_error = np.mean(posterior_sd, axis=0) / np.sqrt(ess)
    scaled_error = np.abs(posterior_mean - true_param) / avg_monte_carlo_standard_error
    np.testing.assert_array_less(scaled_error, stats.norm.ppf(1 - p_val))


def test_nuts_mcse(p_val=0.01):

    loc = np.array([0.0, 3.0])
    scale = np.array([1.0, 2.0])
    rho = np.array(0.75)

    cov = np.diag(scale ** 2)
    cov[0, 1] = rho * scale[0] * scale[1]
    cov[1, 0] = rho * scale[0] * scale[1]

    loc_tt = at.as_tensor(loc)
    scale_tt = at.as_tensor(scale)
    cov_tt = at.as_tensor(cov)

    srng = at.random.RandomStream(seed=0)
    Y_rv = srng.multivariate_normal(loc_tt, cov_tt)

    def logprob_fn(y):
        return joint_logprob({Y_rv: y})

    kernel = nuts.kernel(
        srng,
        logprob_fn,
        scale_tt,
    )

    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    trajectory, updates = aesara.scan(
        kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            None,
            None,
            None,
            None,
        ],
        non_sequences=0.5,
        n_steps=2000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory[0], updates=updates)

    rng = np.random.default_rng()
    posterior_samples = trajectory_generator(rng.standard_normal(2))[-1000:]

    posterior_delta = posterior_samples - loc
    posterior_variance = posterior_delta ** 2
    posterior_correlation = np.prod(posterior_delta, axis=-1, keepdims=True) / (
        scale[0] * scale[1]
    )

    assert_mcse(posterior_samples, loc)
    assert_mcse(posterior_variance, scale ** 2)
    assert_mcse(posterior_correlation, rho)


def test_hmc_mcse(p_val=0.01):

    loc = np.array([0.0, 3.0])
    scale = np.array([1.0, 2.0])
    rho = np.array(0.75)

    cov = np.diag(scale ** 2)
    cov[0, 1] = rho * scale[0] * scale[1]
    cov[1, 0] = rho * scale[0] * scale[1]

    loc_tt = at.as_tensor(loc)
    scale_tt = at.as_tensor(scale)
    cov_tt = at.as_tensor(cov)

    srng = at.random.RandomStream(seed=1)
    Y_rv = srng.multivariate_normal(loc_tt, cov_tt)

    def logprob_fn(y):
        return joint_logprob({Y_rv: y})

    kernel = hmc.kernel(srng, logprob_fn, scale_tt, at.as_tensor(100))

    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    trajectory, updates = aesara.scan(
        kernel,
        outputs_info=[
            {"initial": initial_state[0]},
            {"initial": initial_state[1]},
            {"initial": initial_state[2]},
            None,
        ],
        non_sequences=0.5,
        n_steps=5000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory[0], updates=updates)

    rng = np.random.default_rng()
    posterior_samples = trajectory_generator(rng.standard_normal(2))[-1000:]

    posterior_delta = posterior_samples - loc
    posterior_variance = posterior_delta ** 2
    posterior_correlation = np.prod(posterior_delta, axis=-1, keepdims=True) / (
        scale[0] * scale[1]
    )

    assert_mcse(posterior_samples, loc)
    assert_mcse(posterior_variance, scale ** 2)
    assert_mcse(posterior_correlation, rho)
