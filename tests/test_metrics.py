import aesara
import aesara.tensor as aet
import numpy as np
import pytest
from aesara.tensor.random.utils import RandomStream

from aehmc.metrics import gaussian_metric

momentum_test_cases = [
    (1.0, 0.144),
    (np.array([1.0]), np.array([0.144])),
    (np.array([1.0, 1.0]), np.array([0.144, 1.27])),
    (np.array([[1.0, 0], [0, 1.0]]), np.array([0.144, 1.27])),
]


@pytest.mark.parametrize("case", momentum_test_cases)
def test_gaussian_metric_momentum(case):

    inverse_mass_matrix_val, expected_momentum = case

    # Momentum
    momentum_fn, _, _ = gaussian_metric(aet.as_tensor(inverse_mass_matrix_val))
    srng = RandomStream(seed=59)
    momentum_generator = aesara.function([], momentum_fn(srng))
    generated_momentum = momentum_generator()

    assert np.shape(generated_momentum) == np.shape(expected_momentum)
    assert generated_momentum == pytest.approx(expected_momentum, 1e-2)


kinetic_energy_test_cases = [
    (np.array([1.0]), np.array([1.0]), 0.5),
    (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0),
    (np.array([[1.0, 0], [0, 1.0]]), np.array([1.0, 1.0]), 1.0),
]


@pytest.mark.parametrize("case", kinetic_energy_test_cases)
def test_gaussian_metric_kinetic_energy(case):

    inverse_mass_matrix_val, momentum_val, expected_energy = case

    if inverse_mass_matrix_val.ndim == 1:
        inverse_mass_matrix = aet.vector("inverse_mass_matrix")
    else:
        inverse_mass_matrix = aet.matrix("inverse_mass_matrix")

    _, kinetic_energy_fn, _ = gaussian_metric(inverse_mass_matrix)
    momentum = aet.vector("momentum")
    kinetic_energy = aesara.function(
        (inverse_mass_matrix, momentum), kinetic_energy_fn(momentum)
    )

    assert kinetic_energy(inverse_mass_matrix_val, momentum_val) == expected_energy


turning_test_cases = [np.array([1.0, 1.0]), np.array([[1.0, 0.0], [0.0, 1.0]])]


@pytest.mark.parametrize("inverse_mass_matrix_val", turning_test_cases)
def test_turning(inverse_mass_matrix_val):

    if inverse_mass_matrix_val.ndim == 1:
        inverse_mass_matrix = aet.vector("inverse_mass_matrix")
    else:
        inverse_mass_matrix = aet.matrix("inverse_mass_matrix")

    _, _, turning_fn = gaussian_metric(inverse_mass_matrix)

    p_left = aet.vector("p_left")
    p_right = aet.vector("p_right")
    p_sum = aet.vector("p_sum")
    is_turning_fn = aesara.function(
        (inverse_mass_matrix, p_left, p_right, p_sum),
        turning_fn(p_left, p_right, p_sum),
    )

    n_dim = np.shape(inverse_mass_matrix_val)[0]
    is_turning = is_turning_fn(
        inverse_mass_matrix_val, np.ones(n_dim), np.ones(n_dim), np.ones(n_dim)
    ).item()
    assert is_turning is True


def test_fail_wrong_mass_matrix_dimension():
    inverse_mass_matrix = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    with pytest.raises(ValueError):
        _ = gaussian_metric(inverse_mass_matrix)
