import numpy as np
import aesara
import aesara.tensor as aet
from aesara_hmc.metrics import gaussian_metric
from aesara.tensor.random.utils import RandomStream
import pytest

test_cases = [
    (np.array([1.0]), -0.218),
    (np.array([1.0, 1.0]), np.array([-0.218, 0.268])),
    (np.array([[1.0, 0], [0, 1.0]]), np.array([-0.218, 0.268])),
]


@pytest.mark.parametrize("case", test_cases)
def test_gaussian_metric(case):

    inverse_mass_matrix, expected_momentum = case

    momentum_fn, _, _ = gaussian_metric(inverse_mass_matrix)

    srng = RandomStream(seed=59)
    momentum_generator = aesara.function([], momentum_fn(srng))

    assert momentum_generator() == pytest.approx(expected_momentum, 1e-2)
