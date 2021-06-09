"""Test dynamic termination criteria."""
import pytest
import aesara
import aesara.tensor as aet
import numpy as np

from aehmc.termination import iterative_uturn
from aehmc.metrics import gaussian_metric


@pytest.mark.parametrize(
    "checkpoint_idxs, expected_turning",
    [((3, 2), False), ((3, 3), True), ((0, 0), False), ((0, 1), True), ((1, 3), True)],
)
def test_iterative_turning_termination(checkpoint_idxs, expected_turning):
    inverse_mass_matrix = aet.as_tensor(np.ones(1))
    _, _, is_turning = gaussian_metric(inverse_mass_matrix)
    _, _, is_iterative_turning = iterative_uturn(is_turning)

    momentum = aet.scalar("momentum")
    momentum_sum = aet.scalar("momentum_sum")

    idx_min, idx_max = checkpoint_idxs
    idx_min = aet.as_tensor(idx_min)
    idx_max = aet.as_tensor(idx_max)
    momentum_ckpts = aet.as_tensor(np.array([1.0, 2.0, 3.0, -2.0]))
    momentum_sum_ckpts = aet.as_tensor(np.array([2.0, 4.0, 4.0, -1.0]))
    ckpt_state = (momentum_ckpts, momentum_sum_ckpts, idx_min, idx_max)

    _, _, is_iterative_turning_fn = iterative_uturn(is_turning)
    is_iterative_turning = is_iterative_turning_fn(ckpt_state, momentum_sum, momentum)
    fn = aesara.function((momentum, momentum_sum), is_iterative_turning)

    actual_turning = fn(1., 3.)

    print(actual_turning)

    assert expected_turning == actual_turning[1][-1]
