import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream

from aesara_hmc.metrics import gaussian_metric

momentum_test_cases = [
    (np.array([1.0]), -0.218),
    (np.array([1.0, 1.0]), np.array([-0.218, 0.268])),
    (np.array([[1.0, 0], [0, 1.0]]), np.array([-0.218, 0.268])),
]


@pytest.mark.parametrize("case", momentum_test_cases)
def test_gaussian_metric_momentum(case):

    inverse_mass_matrix, expected_momentum = case

    # Momentum
    momentum_fn, _, _ = gaussian_metric(inverse_mass_matrix)
    srng = RandomStream(seed=59)
    momentum_generator = aesara.function([], momentum_fn(srng))
    assert momentum_generator() == pytest.approx(expected_momentum, 1e-2)


kinetic_energy_test_cases = [
    (np.array([1.0]), np.array([1.0]), 0.5),
    (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0),
    (np.array([[1.0, 0], [0, 1.0]]), np.array([1.0, 1.0]), 1.0),
]


@pytest.mark.parametrize("case", kinetic_energy_test_cases)
def test_gaussian_metric_kinetic_energy(case):

    inverse_mass_matrix, momentum_val, expected_energy = case

    _, kinetic_energy_fn, _ = gaussian_metric(inverse_mass_matrix)
    momentum = aet.vector("momentum")
    kinetic_energy = aesara.function((momentum,), kinetic_energy_fn(momentum))

    assert kinetic_energy(momentum_val) == expected_energy


turning_test_cases = [np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])]


@pytest.mark.parametrize("inverse_mass_matrix", turning_test_cases)
def test_turning(inverse_mass_matrix):

    _, _, turning_fn = gaussian_metric(inverse_mass_matrix)

    p_left = aet.vector("p_left")
    p_right = aet.vector("p_right")
    p_sum = aet.vector("p_sum")
    is_turning_fn = aesara.function(
        (p_left, p_right, p_sum), turning_fn(p_left, p_right, p_sum)
    )

    n_dim = inverse_mass_matrix.ndim
    is_turning = is_turning_fn(np.ones(n_dim), np.ones(n_dim), np.ones(n_dim)).item()
    assert is_turning is True
