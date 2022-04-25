import aesara
import aesara.tensor as at
import arviz
import numpy as np
import pytest
import scipy.stats as stats
from aeppl import joint_logprob
from aesara.tensor.var import TensorVariable

from aehmc import hmc, nuts, window_adaptation


def test_warmup_scalar():
    """Test the warmup on a univariate normal distribution."""

    srng = at.random.RandomStream(seed=0)
    Y_rv = srng.normal(1, 2)

    def logprob_fn(y: TensorVariable):
        logprob = joint_logprob({Y_rv: y})
        return logprob

    def kernel_factory(inverse_mass_matrix: TensorVariable):
        return nuts.kernel(
            srng,
            logprob_fn,
            inverse_mass_matrix,
        )

    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    state, (step_size, inverse_mass_matrix), updates = window_adaptation.run(
        kernel_factory, initial_state, num_steps=1000
    )

    # Compile the warmup and execute to get a value for the step size and the
    # mass matrix.
    warmup_fn = aesara.function(
        (y_vv,),
        (state[0], state[1], state[2], step_size, inverse_mass_matrix),
        updates=updates,
    )

    *final_state, step_size, inverse_mass_matrix = warmup_fn(3.0)

    assert final_state[0] != 3.0  # the chain has moved
    assert np.ndim(step_size) == 0  # scalar step size
    assert step_size > 0.1 and step_size < 2  # stable step size
    assert np.ndim(inverse_mass_matrix) == 0  # scalar mass matrix
    assert inverse_mass_matrix == pytest.approx(4, rel=1.0)


def test_warmup_vector():
    """Test the warmup on a multivariate normal distribution."""

    loc = np.array([0.0, 3.0])
    scale = np.array([1.0, 2.0])
    cov = np.diag(scale**2)

    srng = at.random.RandomStream(seed=0)
    Y_rv = srng.multivariate_normal(loc, cov)

    def logprob_fn(y: TensorVariable):
        logprob = joint_logprob({Y_rv: y})
        return logprob

    def kernel_factory(inverse_mass_matrix: TensorVariable):
        return nuts.kernel(
            srng,
            logprob_fn,
            inverse_mass_matrix,
        )

    y_vv = Y_rv.clone()
    initial_state = nuts.new_state(y_vv, logprob_fn)

    state, (step_size, inverse_mass_matrix), updates = window_adaptation.run(
        kernel_factory, initial_state, num_steps=1000
    )

    # Compile the warmup and execute to get a value for the step size and the
    # mass matrix.
    warmup_fn = aesara.function(
        (y_vv,),
        (state[0], state[1], state[2], step_size, inverse_mass_matrix),
        updates=updates,
    )

    *final_state, step_size, inverse_mass_matrix = warmup_fn([1.0, 1.0])

    assert np.all(final_state[0] != np.array([1.0, 1.0]))  # the chain has moved
    assert np.ndim(step_size) == 0  # scalar step size
    assert step_size > 0.1 and step_size < 2
    assert np.ndim(inverse_mass_matrix) == 1  # scalar mass matrix
    np.testing.assert_allclose(inverse_mass_matrix, scale**2, rtol=1.0)


@pytest.mark.skip(reason="this test is flaky")
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


@pytest.mark.skip(reason="this test is flaky")
def test_nuts():
    """Test the NUTS kernel on a gaussian target."""
    step_size = 0.1
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


def compute_ess(samples):
    d = arviz.convert_to_dataset(np.expand_dims(samples, axis=0))
    ess = arviz.ess(d).to_array().to_numpy().squeeze()
    return ess


def compute_mcse(x):
    ess = compute_ess(x)
    std_x = np.std(x, axis=0, ddof=1)
    return np.mean(x, axis=0), std_x / np.sqrt(ess)


def multivariate_normal_model(srng):
    loc = np.array([0.0, 3.0])
    scale = np.array([1.0, 2.0])
    rho = np.array(0.5)

    cov = np.diag(scale**2)
    cov[0, 1] = rho * scale[0] * scale[1]
    cov[1, 0] = rho * scale[0] * scale[1]

    loc_tt = at.as_tensor(loc)
    cov_tt = at.as_tensor(cov)

    Y_rv = srng.multivariate_normal(loc_tt, cov_tt)

    def logprob_fn(y):
        return joint_logprob({Y_rv: y})

    return (loc, cov, scale, rho), Y_rv, logprob_fn


def test_hmc_mcse():
    """This examples is recommanded in the Stan documentation [1]_ to find bugs
    that introduce bias in the average as well as the variance.

    The example is simple enough to be analytically tractable, but complex enough
    to find subtle bugs.

    If we use the covariance of the normal distribution as the inverse mass matrix
    it can be shown that the dynamics is equivalent to that of two independent
    position variables with variances close to one [2]_ (section 4.1)

    We can also show that, in these circumstances, choosing any step size that
    is smaller than 2. will be stable [2]_ (section 4.2); We adjusted the number
    of integration steps manually in order to get a reasonable number of effective samples.

    .. [1]: https://github.com/stan-dev/stan/wiki/Testing:-Samplers

    See this issue where testing samplers is discussed:
    https://github.com/stan-dev/stan/issues/318
    """
    srng = at.random.RandomStream(seed=1)
    (loc, cov, scale, rho), Y_rv, logprob_fn = multivariate_normal_model(srng)

    L = 100
    inverse_mass_matrix = at.as_tensor(cov)
    kernel = hmc.kernel(srng, logprob_fn, inverse_mass_matrix, L)

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
        non_sequences=1.9,
        n_steps=10000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory, updates=updates)

    rng = np.random.default_rng(seed=0)
    trace = trajectory_generator(rng.standard_normal(2))
    samples = trace[0][1000:]

    # MCSE on the location
    delta_loc = samples - loc
    mean, mcse = compute_mcse(delta_loc)
    p_greater_error = stats.norm.sf(np.abs(mean) / mcse)
    np.testing.assert_array_less(0.01, p_greater_error)

    # MCSE on the variance
    delta_var = np.square(samples - loc) - scale**2
    mean, mcse = compute_mcse(delta_var)
    p_greater_error = stats.norm.sf(np.abs(mean) / mcse)
    np.testing.assert_array_less(0.01, p_greater_error)

    # MCSE on the correlation
    delta_cor = np.prod(samples - loc, axis=1) / np.prod(scale) - rho
    mean, mcse = compute_mcse(delta_cor)
    p_greater_error = stats.norm.sf(np.abs(mean) / mcse)
    np.testing.assert_array_less(0.01, p_greater_error)


@pytest.mark.xfail(
    reason="The current implementation returns biased samples that are not appropriate to estimate the variance."
)
def test_nuts_mcse():

    srng = at.random.RandomStream(seed=1)
    (loc, cov, scale, rho), Y_rv, logprob_fn = multivariate_normal_model(srng)

    inverse_mass_matrix = at.as_tensor(cov)
    kernel = nuts.kernel(srng, logprob_fn, inverse_mass_matrix)

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
        non_sequences=1.9,
        n_steps=10000,
    )

    trajectory_generator = aesara.function((y_vv,), trajectory, updates=updates)

    rng = np.random.default_rng(seed=0)
    trace = trajectory_generator(rng.standard_normal(2))
    samples = trace[0][1000:]

    # MCSE on the location
    delta_loc = samples - loc
    mean, mcse = compute_mcse(delta_loc)
    ess = compute_ess(delta_loc)
    p_greater_error = stats.norm.sf(np.abs(mean) / mcse)
    np.testing.assert_array_less(0.01, p_greater_error)

    # MCSE on the variance
    delta_var = np.square(samples - loc) - scale**2
    mean, mcse = compute_mcse(delta_var)
    ess = compute_ess(delta_var)
    p_greater_error = stats.norm.sf(np.abs(mean) / mcse)
    print(mean, mcse, ess)
    np.testing.assert_array_less(0.01, p_greater_error)

    # MCSE on the correlation
    delta_cor = np.prod(samples - loc, axis=1) / np.prod(scale) - rho
    mean, mcse = compute_mcse(delta_cor)
    ess = compute_ess(delta_cor)
    p_greater_error = stats.norm.sf(np.abs(mean) / mcse)
    np.testing.assert_array_less(0.01, p_greater_error)
    print(mean, mcse, ess)
