import aesara
import aesara.tensor as at
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
    if np.ndim(inverse_mass_matrix_val) == 0:
        inverse_mass_matrix = at.scalar("inverse_mass_matrix")
    elif np.ndim(inverse_mass_matrix_val) == 1:
        inverse_mass_matrix = at.vector("inverse_mass_matrix")
    else:
        inverse_mass_matrix = at.matrix("inverse_mass_matrix")

    momentum_fn, _, _ = gaussian_metric(inverse_mass_matrix)
    srng = RandomStream(seed=59)
    momentum_generator = aesara.function([inverse_mass_matrix], momentum_fn(srng))

    momentum = momentum_generator(inverse_mass_matrix_val)
    assert np.shape(momentum) == np.shape(expected_momentum)
    assert momentum == pytest.approx(expected_momentum, 1e-2)


kinetic_energy_test_cases = [
    (1.0, 1.0, 0.5),
    (np.array([1.0]), np.array([1.0]), 0.5),
    (np.array([1.0, 1.0]), np.array([1.0, 1.0]), 1.0),
    (np.array([[1.0, 0], [0, 1.0]]), np.array([1.0, 1.0]), 1.0),
]


@pytest.mark.parametrize("case", kinetic_energy_test_cases)
def test_gaussian_metric_kinetic_energy(case):

    inverse_mass_matrix_val, momentum_val, expected_energy = case

    if np.ndim(inverse_mass_matrix_val) == 0:
        inverse_mass_matrix = at.scalar("inverse_mass_matrix")
        momentum = at.scalar("momentum")
    elif np.ndim(inverse_mass_matrix_val) == 1:
        inverse_mass_matrix = at.vector("inverse_mass_matrix")
        momentum = at.vector("momentum")
    else:
        inverse_mass_matrix = at.matrix("inverse_mass_matrix")
        momentum = at.vector("momentum")

    _, kinetic_energy_fn, _ = gaussian_metric(inverse_mass_matrix)
    kinetic_energy = aesara.function(
        (inverse_mass_matrix, momentum), kinetic_energy_fn(momentum)
    )

    kinetic = kinetic_energy(inverse_mass_matrix_val, momentum_val)
    assert np.ndim(kinetic) == 0
    assert kinetic == expected_energy


turning_test_cases = [
    (1.0, 1.0, 1.0, 1.0),
    (
        np.array([1.0, 1.0]),  # inverse mass matrix
        np.array([1.0, 1.0]),  # p_left
        np.array([1.0, 1.0]),  # p_right
        np.array([1.0, 1.0]),  # p_sum
    ),
    (
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
        np.array([1.0, 1.0]),
    ),
]


@pytest.mark.parametrize("case", turning_test_cases)
def test_turning(case):

    inverse_mass_matrix_val, p_left_val, p_right_val, p_sum_val = case

    if np.ndim(inverse_mass_matrix_val) == 0:
        inverse_mass_matrix = at.scalar("inverse_mass_matrix")
        p_left = at.scalar("p_left")
        p_right = at.scalar("p_right")
        p_sum = at.scalar("p_sum")
    elif np.ndim(inverse_mass_matrix_val) == 1:
        inverse_mass_matrix = at.vector("inverse_mass_matrix")
        p_left = at.vector("p_left")
        p_right = at.vector("p_right")
        p_sum = at.vector("p_sum")
    else:
        inverse_mass_matrix = at.matrix("inverse_mass_matrix")
        p_left = at.vector("p_left")
        p_right = at.vector("p_right")
        p_sum = at.vector("p_sum")

    _, _, turning_fn = gaussian_metric(inverse_mass_matrix)

    is_turning_fn = aesara.function(
        (inverse_mass_matrix, p_left, p_right, p_sum),
        turning_fn(p_left, p_right, p_sum),
    )

    is_turning = is_turning_fn(
        inverse_mass_matrix_val, p_left_val, p_right_val, p_sum_val
    )

    assert is_turning.ndim == 0
    assert is_turning.item() is True


def test_fail_wrong_mass_matrix_dimension():
    """`gaussian_metric` should fail when the dimension of the mass matrix is greater than 2."""
    inverse_mass_matrix = np.array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    with pytest.raises(ValueError):
        _ = gaussian_metric(inverse_mass_matrix)
